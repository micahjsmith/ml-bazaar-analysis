import gzip
import io
import os.path

import boto3
import botocore
import botocore.config
import pandas as pd

from piex.explorer import LOGGER, S3PipelineExplorer

class UnsignedS3PipelineExplorer(S3PipelineExplorer):

    def _get_client(self):
        return boto3.client('s3',
            config=botocore.config.Config(signature_version=botocore.UNSIGNED))

    def _get_table(self, table_name):
        LOGGER.info("Downloading %s csv from S3", table_name)
        key = os.path.join('csvs', table_name + '.csv.gz')

        s3 = self._get_client()
        obj = s3.get_object(Bucket=self.bucket, Key=key)

        body_bytes = io.BytesIO(obj['Body'].read())
        gzip_file = gzip.GzipFile(fileobj=body_bytes, mode='rb')

        return pd.read_csv(gzip_file)

    def _get_json(self, folder, pipeline_id):
        key = os.path.join(folder, pipeline_id + '.json.gz')
        s3 = self._get_client()
        obj = s3.get_object(Bucket=self.bucket, Key=key)

        body_bytes = io.BytesIO(obj['Body'].read())
        gzip_file = gzip.GzipFile(fileobj=body_bytes, mode='rb')
        return json.load(gzip_file)
