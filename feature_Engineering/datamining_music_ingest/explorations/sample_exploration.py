# Databricks notebook source
from pyspark.sql.functions import*
from pyspark.sql.types import*


# Event Hubs configuration
EH_NAMESPACE  = "music-streaming"
EH_NAME       = "music-log-stream"



#EH_CONN_STR  = spark.conf.get("connection_string")
EH_CONN_STR = "Endpoint=sb://music-streaming.servicebus.windows.net/;SharedAccessKeyName=ListenPolicy;SharedAccessKey=7iJst6EsyffjI2rkn1cUEQmfkdhIHEcY0+AEhBNcsPE=;EntityPath=music-log-stream"

KAFKA_OPTIONS = {
  "kafka.bootstrap.servers"  : f"{EH_NAMESPACE}.servicebus.windows.net:9093",
  "subscribe"                : EH_NAME,
  "kafka.sasl.mechanism"     : "PLAIN",
  "kafka.security.protocol"  : "SASL_SSL",
  "kafka.sasl.jaas.config"   : f"kafkashaded.org.apache.kafka.common.security.plain.PlainLoginModule required username=\"$ConnectionString\" password=\"{EH_CONN_STR}\";",
  "kafka.request.timeout.ms" : 10000,
  "kafka.session.timeout.ms" : 10000,
  "maxOffsetsPerTrigger"     : 10000,
  "failOnDataLoss"           : 'true',
  "startingOffsets"          : 'earliest'
}




df = spark.readStream.format("kafka")\
            .options(**KAFKA_OPTIONS)\
            .load()

display(df,checkpointLocation = "/Volumes/music/bronze/my_volume/volume_folder/")







