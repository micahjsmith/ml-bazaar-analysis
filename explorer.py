import gzip
import io
import os.path

import boto3
import botocore
import botocore.config
import mit_d3m.db
import pandas as pd

from piex.explorer import MongoPipelineExplorer, S3PipelineExplorer

bucket = 'ml-pipelines-2018'
mongo_config_file = 'mongodb_config.json'


def get_explorer():
    try:
        db = mit_d3m.db.get_db(config=mongo_config_file)
        ex = MongoPipelineExplorer(db)
    except Exception:
        ex = S3PipelineExplorer(bucket)

    return ex
