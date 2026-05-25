# Databricks notebook source
# MAGIC %md
# MAGIC # Xuất dữ liệu cho Web Dashboard
# MAGIC **Mục tiêu:** Gộp thông tin Persona (từ quá trình Gom cụm) và thông tin Churn Risk (từ quá trình Dự đoán) cùng với các feature gốc để tạo ra bảng dữ liệu thống nhất phục vụ hiển thị trên Web Dashboard.
# MAGIC **Output:** Bảng `default.web_dashboard_data` và file CSV (tùy chọn)

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Đọc các bảng dữ liệu

# COMMAND ----------

gold_features_df = spark.table("default.gold_user_features")
persona_df       = spark.table("default.user_persona_results")
churn_pred_df    = spark.table("default.churn_predictions")

print(f"Tổng User trong Gold: {gold_features_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Merge Dữ liệu

# COMMAND ----------

# Join Persona
dashboard_df = gold_features_df.join(persona_df, on="user_id", how="left")

# Join Churn Prediction
dashboard_df = dashboard_df.join(churn_pred_df, on="user_id", how="left")

# Điền các giá trị null (nếu có do left join)
dashboard_df = dashboard_df.fillna({
    "persona_name": "Chưa phân cụm",
    "churn_risk_percent": 0.0,
    "risk_level": "UNKNOWN"
})

display(dashboard_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Xuất kết quả

# COMMAND ----------

# 1. Lưu lại thành Table trên Databricks để kết nối BI Tool (Superset, PowerBI, Tableau)
dashboard_df.write.mode("overwrite").saveAsTable("default.web_dashboard_data_v2")

print("Đã lưu bảng Dashboard thành công: default.web_dashboard_data_v2")

# 2. (Tùy chọn) Xuất ra định dạng CSV nếu Web Frontend/Backend cần đọc file trực tiếp
# DBFS_PATH = "/FileStore/tables/dashboard/web_dashboard_data.csv"
# dashboard_df.repartition(1).write.mode("overwrite").csv(DBFS_PATH, header=True)
# print(f"Đã xuất file CSV ra {DBFS_PATH}")
