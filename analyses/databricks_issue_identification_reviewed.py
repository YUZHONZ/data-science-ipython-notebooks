# Databricks notebook source
# MAGIC %pip install pydantic==2.3.0
# MAGIC %pip install pydantic-settings==2.2.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# DBTITLE 1,Runtime setup and widgets
import os
import sys
import json
from typing import Dict, Any

from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf, from_json, explode, flatten, collect_list, map_from_entries, struct, regexp_extract, array_distinct
from pyspark.sql.types import MapType, StringType, ArrayType, StructType, StructField
from pyspark.sql.window import Window

from databricks.sdk.runtime import dbutils


dbutils.widgets.dropdown(
    "env",
    "local",
    ["local", "dev", "staging", "prod"],
    "Environment Name",
)

dbutils.widgets.dropdown(
    "open_ai_endpoint",
    "default",
    ["default", "alt"],
    "OpenAI Endpoint",
)

# COMMAND ----------
# DBTITLE 1,Environment resolution
env = dbutils.widgets.get("env")
open_ai_endpoint = dbutils.widgets.get("open_ai_endpoint") or "default"

ENV_TO_DB_ENV = {
    "local": "dev",
    "dev": "dev",
    "staging": "stg",
    "prod": "prd",
}
db_env = ENV_TO_DB_ENV.get(env, env)

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
notebook_path = "/Workspace" + os.path.dirname(notebook_path)
sys.path.append(f"{notebook_path}/../src")

# COMMAND ----------
# DBTITLE 1,Project imports and helpers
from helpers.data_helpers import DBHelper
from common_functions import transform_questions
from libs.utils import get_current_time, load_json_file
from libs.document_stage_status import DocumentStageStatus
from helpers.prompt_helper import PromptHelper
from helpers.llm_task_helper import LLMTaskHelper
from helpers.azureopenaimodel import AzureOpenAiModel
from libs.settings import get_settings
from libs.openai_api import OpenaiApi
from libs.cost_calculator import CostCalculator

# COMMAND ----------
# DBTITLE 1,Init database and prompt helpers
db_helper = DBHelper(env=db_env, path=notebook_path)
document_stage_status = DocumentStageStatus(dbh=db_helper)

catalog_name = db_helper.config.get("catalog_name")
schema_name = db_helper.config.get("schema_name")
schema_config = db_helper.schema_config

llm_task_helper = LLMTaskHelper(db_env, db_helper.path)
prompt_helper = PromptHelper(db_helper, llm_task_helper)
openai_api = OpenaiApi()

for table_name in ["llm_prompt_template"]:
    full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
    db_helper.create_table_if_not_exists(full_table_name, schema_config[table_name])

# COMMAND ----------
# DBTITLE 1,Model and settings setup
os.environ["OPENAI_API_VERSION"] = "2025-04-01-preview"
os.environ["DESTINATION"] = env
os.environ["OPENAI_API_TYPE"] = "gpt-4o"

if open_ai_endpoint == "alt":
    os.environ["AZURE_ENDPOINT_ALT"] = "https://openai-00002-non-prod-1.openai.azure.com/"
    os.environ["OPENAI_API_KEY_ALT"] = dbutils.secrets.get(scope="openai_00002_1", key="apikey")
else:
    # Placeholder/default endpoint wiring can be updated to your real default scope.
    os.environ["AZURE_ENDPOINT_DEFAULT"] = ""
    os.environ["OPENAI_API_KEY_DEFAULT"] = ""

settings = get_settings()

gpt5_mini_model = AzureOpenAiModel(
    api_url=settings.AZURE_ENDPOINT_ALT,
    api_key=settings.OPENAI_API_KEY_ALT,
    api_version=settings.OPENAI_API_VERSION,
    api_type=settings.OPENAI_API_TYPE,
    deployment_name="gpt-5-mini-global",
)

gpt5_model = AzureOpenAiModel(
    api_url=settings.AZURE_ENDPOINT_ALT,
    api_key=settings.OPENAI_API_KEY_ALT,
    api_version=settings.OPENAI_API_VERSION,
    api_type=settings.OPENAI_API_TYPE,
    deployment_name="gpt-5-global",
)

gpt4o_model = AzureOpenAiModel(
    api_url=settings.AZURE_ENDPOINT_ALT,
    api_key=settings.OPENAI_API_KEY_ALT,
    api_version=settings.OPENAI_API_VERSION,
    api_type=settings.OPENAI_API_TYPE,
    deployment_name="gpt-4o",
)

model_dict = {
    "gpt5_mini": gpt5_mini_model,
    "gpt5": gpt5_model,
    "gpt4o": gpt4o_model,
}

# COMMAND ----------
# DBTITLE 1,Load base input data
RUN_ID = "nearmap_3"

df = db_helper.fetch_table(table_name="hq_document_metadata", filter_column="run_id", filter_value=RUN_ID)
df = df.withColumnRenamed("file_name", "doc_id").withColumnRenamed("file_path", "pdf_path")

df_builder_10 = df.filter(F.col("report_subtype") == "builder_report").limit(9)
df_desktop_10 = df.filter(F.col("report_subtype") == "desktop_report").limit(14)
df = df_builder_10.union(df_desktop_10)

questions_path = "../config/questions_extract/assessment_questions.json"
questions = load_json_file(questions_path)

# COMMAND ----------
# DBTITLE 1,Issue-identification input prep: selected question-answer map
doc_extraction_df = df.filter(~F.col("report_subtype").isin("theft_new_claim_report"))

subtypes = [r["report_subtype"] for r in doc_extraction_df.select("report_subtype").distinct().collect()]

report_question_list = {}
for report_subtype in subtypes:
    key_questions = questions.get(f"{report_subtype}_questions")
    _, _, original_question_list = transform_questions(key_questions, report_subtype)
    report_question_list[report_subtype] = original_question_list or []

bcast_questions = spark.sparkContext.broadcast(report_question_list)

@F.udf(MapType(StringType(), StringType()))
def feature_selection_udf(extracted_data_json: str, report_subtype: str):
    try:
        data = extracted_data_json if isinstance(extracted_data_json, dict) else json.loads(extracted_data_json) if extracted_data_json else {}
        if not isinstance(data, dict):
            return {}

        keys = bcast_questions.value.get(report_subtype, [])
        out = {}
        for k in keys:
            v = data.get(k)
            if v is not None:
                out[str(k)] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        return out
    except Exception:
        return {}

doc_extraction_df_spark = (
    doc_extraction_df
    .withColumn("extracted_data_selected", feature_selection_udf(F.col("extracted_data"), F.col("report_subtype")))
    .orderBy(F.col("report_type").asc())
    .withColumn("extracted_data_selected", F.to_json(F.col("extracted_data_selected")))
)

# COMMAND ----------
# DBTITLE 1,Issue-identification input prep: question-to-pages map
pdf_paths = [row["pdf_path"] for row in df.select("pdf_path").distinct().collect()]

filtered_df = spark.sql(
    """
    SELECT *
    FROM datascience_prd_silver.ds_doc_extraction_home_claims.document_extraction_complete_logs
    """
).filter(col("PDF_Path").isin(pdf_paths))

window_spec = Window.partitionBy("PDF_Path")
filtered_df = filtered_df.withColumn("latest_run_datetime", F.max("run_datetime").over(window_spec))
filtered_df = filtered_df.filter(F.col("run_datetime") == F.col("latest_run_datetime")).drop("latest_run_datetime")

def extract_questions_and_pages(model_response: Dict[str, Any]):
    if not model_response:
        return {}
    combined = {}
    for question, details in model_response.items():
        page_number = details.get("page_number", "") if isinstance(details, dict) else ""
        if page_number is not None:
            combined.setdefault(question, []).append(str(page_number))
            combined[question] = sorted(combined[question])
    return combined

extract_questions_and_pages_udf = udf(extract_questions_and_pages, MapType(StringType(), ArrayType(StringType())))

filtered_df = filtered_df.withColumn(
    "model_response_parsed",
    from_json(col("model_response"), MapType(StringType(), MapType(StringType(), StringType()))),
).withColumn("questions_and_pages", extract_questions_and_pages_udf(col("model_response_parsed")))

question_page_df = (
    filtered_df
    .select(col("pdf_path"), explode(col("questions_and_pages")).alias("question", "page_numbers"))
    .groupBy("pdf_path", "question")
    .agg(array_distinct(flatten(collect_list("page_numbers"))).alias("page_numbers"))
    .groupBy("pdf_path")
    .agg(map_from_entries(collect_list(struct(col("question"), col("page_numbers")))).alias("questions_and_pages"))
)

question_page_df = question_page_df.withColumn("doc_id", regexp_extract(col("pdf_path"), r".*/([^/]+\.pdf)$", 1))

doc_extraction_df_spark = doc_extraction_df_spark.join(
    question_page_df.select("pdf_path", "questions_and_pages"),
    on="pdf_path",
    how="left",
)

# COMMAND ----------
# DBTITLE 1,Create question-answer structure for prompt
result_schema = ArrayType(
    StructType([
        StructField("Question", StringType(), True),
        StructField("Question_Page_No", ArrayType(StringType()), True),
        StructField("Answer", StringType(), True),
    ])
)

def generate_question_answer_structure(questions_and_pages, extracted_data_selected):
    if not questions_and_pages or not extracted_data_selected:
        return []
    return [
        {"Question": q, "Question_Page_No": questions_and_pages.get(q, []), "Answer": a}
        for q, a in extracted_data_selected.items()
    ]

doc_extraction_df_spark = doc_extraction_df_spark.withColumn(
    "extracted_data_selected_parsed",
    from_json(col("extracted_data_selected"), MapType(StringType(), StringType())),
)

generate_question_answer_structure_udf = udf(generate_question_answer_structure, result_schema)
doc_extraction_df_spark = doc_extraction_df_spark.withColumn(
    "question_answer_structure",
    generate_question_answer_structure_udf(col("questions_and_pages"), col("extracted_data_selected_parsed")),
)

# COMMAND ----------
# DBTITLE 1,Prompts
prompts = prompt_helper.get_prompt_templates(identifier="issue_identification")
user_prompt = prompts[prompts["role_type"] == "user"].sort_values(by="version", ascending=False).iloc[0].prompt_text
system_prompt = prompts[prompts["role_type"] == "system"].sort_values(by="version", ascending=False).iloc[0].prompt_text

EVIDENCE_EXAMPLE = '{"question":"<verbatim>","question_page_no": "<verbatim>","quote": <verbatim>}'

doc_extraction_df_spark = (
    doc_extraction_df_spark
    .withColumn("System_Prompt", F.lit(system_prompt))
    .withColumn(
        "User_Prompt",
        F.format_string(
            user_prompt,
            F.col("final_loss_cause"),
            F.to_json(F.col("question_answer_structure")),
            F.lit(EVIDENCE_EXAMPLE),
        ),
    )
    .filter(F.col("questions_and_pages").isNotNull())
)

# COMMAND ----------
# DBTITLE 1,Issue extraction runner
proxy_secret_scope = "nginx_proxy_sp"
sp_client_id = dbutils.secrets.get(scope=proxy_secret_scope, key="client_id")
proxy_client_secret = dbutils.secrets.get(scope=proxy_secret_scope, key="client_secret")
dbutils_secret = (sp_client_id, proxy_client_secret)

class IssueIdentification:
    @staticmethod
    def run_extraction(source_df, model=None, dbutils_secret=None):
        openai_udf = openai_api.get_openai_pandas_udf(
            model,
            max_tokens=4096,
            retries=5,
            initial_batch_size=8,
            dbutils_secret=dbutils_secret,
        )

        openai_df = source_df.withColumn(
            "OpenAI_Response",
            openai_udf(F.col("claim_number"), F.col("doc_id"), F.col("System_Prompt"), F.col("User_Prompt")),
        )

        openai_df = openai_df.select(
            "claim_number",
            "doc_id",
            "extracted_data_selected",
            F.col("OpenAI_Response.System_Prompt").alias("System_Prompt"),
            F.col("OpenAI_Response.User_Prompt").alias("User_Prompt"),
            F.col("OpenAI_Response.response").alias("Model_Response"),
            F.col("OpenAI_Response.raw_response").alias("Raw_Model_Response"),
            F.col("OpenAI_Response.input_tokens").alias("Input_Tokens"),
            F.col("OpenAI_Response.output_tokens").alias("Output_Tokens"),
            F.col("OpenAI_Response.error").alias("API_Error"),
            F.col("OpenAI_Response.latency").alias("latency"),
            F.col("OpenAI_Response.retry_attempt").alias("api_retry_attempt"),
        )

        return openai_df.withColumn("Cost", CostCalculator.cost_pandas_udf(F.col("Input_Tokens"), F.col("Output_Tokens")))

# COMMAND ----------
# DBTITLE 1,Run extraction
issue_identification = IssueIdentification()
doc_extraction_df_spark = doc_extraction_df_spark.repartition(16)

final_df = None
for model_name, model in model_dict.items():
    df_result = issue_identification.run_extraction(doc_extraction_df_spark, model, dbutils_secret)
    df_result = df_result.withColumn("run_datetime", F.lit(get_current_time())).withColumn("run_id", F.lit(model_name))
    final_df = df_result if final_df is None else final_df.unionByName(df_result)

final_df = final_df.cache()
final_df.count()

# COMMAND ----------
# DBTITLE 1,Parse and flatten model output

evidence_schema = ArrayType(StructType([
    StructField("quote", StringType()),
    StructField("question", StringType()),
    StructField("question_page_no", ArrayType(StringType())),
]))

ncrd_explanation_schema = ArrayType(StructType([
    StructField("quote", StringType()),
    StructField("question", StringType()),
    StructField("question_page_no", ArrayType(StringType())),
    StructField("reasoning", StringType()),
]))

location_schema = StructType([
    StructField("zone", StringType()),
    StructField("roof_section", StringType()),
    StructField("room", StringType()),
    StructField("level", StringType()),
    StructField("detail", StringType()),
])

component_schema = StructType([
    StructField("standard_component_name", StringType()),
    StructField("component_detailed_name", StringType()),
    StructField("component_location", StringType()),
    StructField("parent_component_name", StringType()),
])

row_schema = ArrayType(StructType([
    StructField("claim_number", StringType()),
    StructField("component", StringType()),
    StructField("component_normalized", component_schema),
    StructField("issue", StringType()),
    StructField("result_of_issue", StringType()),
    StructField("type_of_issue", StringType()),
    StructField("cause_of_damage", StringType()),
    StructField("related_to_primary_cause", StringType()),
    StructField("relation_to_primary_cause", StringType()),
    StructField("related_to_primary_cause_explanation", StringType()),
    StructField("ncrd", StringType()),
    StructField("ncrd_explanation", ncrd_explanation_schema),
    StructField("ncrd_required", StringType()),
    StructField("ncrd_required_explanation", ncrd_explanation_schema),
    StructField("ncrd_recommended", StringType()),
    StructField("ncrd_recommended_explanation", ncrd_explanation_schema),
    StructField("report_evidence", evidence_schema),
    StructField("location_free_text", StringType()),
    StructField("location_normalized", location_schema),
]))

parsed_rows_df = final_df.withColumn(
    "rows",
    F.from_json(
        F.when(F.col("Model_Response").rlike(r"^\s*\["), F.col("Model_Response")).otherwise(
            F.concat(F.lit("["), F.col("Model_Response"), F.lit("]"))
        ),
        row_schema,
    ),
)

issues_df = (
    parsed_rows_df
    .withColumn("entry", F.explode_outer("rows"))
    .select(
        F.coalesce(F.col("claim_number"), F.col("entry.claim_number")).alias("claim_number"),
        "doc_id",
        "run_id",
        "run_datetime",
        F.col("entry.component").alias("component"),
        F.col("entry.component_normalized").alias("component_normalized"),
        F.col("entry.issue").alias("issue"),
        F.col("entry.result_of_issue").alias("result_of_issue"),
        F.col("entry.type_of_issue").alias("type_of_issue"),
        F.col("entry.cause_of_damage").alias("cause_of_damage"),
        F.col("entry.related_to_primary_cause").alias("related_to_primary_cause"),
        F.col("entry.relation_to_primary_cause").alias("relation_to_primary_cause"),
        F.col("entry.related_to_primary_cause_explanation").alias("related_to_primary_cause_explanation"),
        F.col("entry.ncrd").alias("ncrd"),
        F.col("entry.ncrd_explanation").alias("ncrd_explanation"),
        F.col("entry.ncrd_required").alias("ncrd_required"),
        F.col("entry.ncrd_required_explanation").alias("ncrd_required_explanation"),
        F.col("entry.ncrd_recommended").alias("ncrd_recommended"),
        F.col("entry.ncrd_recommended_explanation").alias("ncrd_recommended_explanation"),
        F.col("entry.report_evidence").alias("report_evidence"),
        F.col("entry.location_free_text").alias("location_free_text"),
        F.col("entry.location_normalized").alias("location_normalized"),
    )
)

final_issues_df = (
    issues_df
    .groupBy(
        "claim_number", "doc_id", "run_id", "component", "component_normalized", "issue", "result_of_issue", "type_of_issue",
        "cause_of_damage", "related_to_primary_cause", "relation_to_primary_cause", "related_to_primary_cause_explanation",
        "ncrd", "ncrd_explanation", "ncrd_required", "ncrd_required_explanation", "ncrd_recommended",
        "ncrd_recommended_explanation", "location_free_text", "location_normalized",
    )
    .agg(flatten(collect_list("report_evidence")).alias("report_evidence_all"))
    .withColumn("standard_component_name", F.col("component_normalized.standard_component_name"))
    .withColumn("component_detailed_name", F.col("component_normalized.component_detailed_name"))
    .withColumn("component_location", F.col("component_normalized.component_location"))
    .withColumn("parent_component_name", F.col("component_normalized.parent_component_name"))
    .withColumn("location_zone", F.col("location_normalized.zone"))
    .withColumn("location_roof_section", F.col("location_normalized.roof_section"))
    .withColumn("location_room", F.col("location_normalized.room"))
    .withColumn("location_level", F.col("location_normalized.level"))
    .withColumn("location_detail", F.col("location_normalized.detail"))
    .withColumn("location_normalized_json", F.to_json("location_normalized"))
    .withColumn("report_evidence_json", F.to_json("report_evidence_all"))
    .drop("component_normalized", "location_normalized")
    .withColumn("run_datetime", F.coalesce(F.col("run_datetime"), F.lit(get_current_time())))
    .withColumn("issue_id", F.expr("uuid()"))
)

# COMMAND ----------
# DBTITLE 1,Issue/NCRD summary by model (run_id)
issue_counts_long_df = (
    final_issues_df
    .groupBy("doc_id", "run_id")
    .agg(
        F.count("issue_id").alias("number_of_issues"),
        F.sum(F.when(F.col("ncrd_required") == "Yes", F.lit(1)).otherwise(F.lit(0))).alias("number_of_ncrd_required"),
        F.sum(F.when(F.col("ncrd_recommended") == "Yes", F.lit(1)).otherwise(F.lit(0))).alias("number_of_ncrd_recommended"),
    )
)

issue_count_wide_df = (
    issue_counts_long_df
    .groupBy("doc_id")
    .pivot("run_id")
    .agg(F.first("number_of_issues"))
    .fillna(0)
)
for c in issue_count_wide_df.columns:
    if c != "doc_id":
        issue_count_wide_df = issue_count_wide_df.withColumnRenamed(c, f"{c}_number_of_issues")

ncrd_required_wide_df = (
    issue_counts_long_df
    .groupBy("doc_id")
    .pivot("run_id")
    .agg(F.first("number_of_ncrd_required"))
    .fillna(0)
)
for c in ncrd_required_wide_df.columns:
    if c != "doc_id":
        ncrd_required_wide_df = ncrd_required_wide_df.withColumnRenamed(c, f"{c}_number_of_ncrd_required")

ncrd_recommended_wide_df = (
    issue_counts_long_df
    .groupBy("doc_id")
    .pivot("run_id")
    .agg(F.first("number_of_ncrd_recommended"))
    .fillna(0)
)
for c in ncrd_recommended_wide_df.columns:
    if c != "doc_id":
        ncrd_recommended_wide_df = ncrd_recommended_wide_df.withColumnRenamed(c, f"{c}_number_of_ncrd_recommended")

issue_counts_wide_df = (
    issue_count_wide_df
    .join(ncrd_required_wide_df, on="doc_id", how="left")
    .join(ncrd_recommended_wide_df, on="doc_id", how="left")
)

# COMMAND ----------
# DBTITLE 1,Write output table
table_name = f"{catalog_name}.{schema_name}.hq_llm_issue_identification_results_ncrd_expanded"
table_schema = schema_config.get("hq_llm_issue_identification_results_ncrd_required_recommended_flattened")

db_helper.append_to_database_spark(
    sdf=final_issues_df,
    table_name=table_name,
    table_schema=table_schema,
)
