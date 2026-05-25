import dlt
from pyspark.sql.types import *

spark.conf.set(
    "fs.azure.account.key.musicprojectdm.dfs.core.windows.net", 
    "FR9y9JaINWhWWrSAdOu4FFlaJc3RedM6P9CDHgB2YemEYdzal6I62O3DmZyEGrjUIPkWp7FcBwl4+AStZ0hmwg=="
)

music_schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("artist_name", StringType(), True),
    StructField("recording_msid", StringType(), True),
    StructField("track_name", StringType(), True),      
    StructField("release_name", StringType(), True)     
])

@dlt.table(
    name="live_music_logs",
    comment="Bảng Bronze lưu trữ lịch sử nghe nhạc Real-time (6 trường) từ Container bronzelive"
)
def live_music_raw():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .schema(music_schema)
        # Sửa tên container thành bronzelive
        .load("abfss://bronzelive@musicprojectdm.dfs.core.windows.net/")
    )