import sagemaker
from sagemaker.sklearn.estimator import SKLearn

session = sagemaker.Session()
role = sagemaker.get_execution_role()

estimator = SKLearn(
    entry_point="train.py",
    source_dir=".",
    role=role,
    instance_type="ml.m5.large",
    framework_version="1.2-1",
    py_version="py3",
)

estimator.fit({
    "train": "s3://projectcontinue/data/train/BostonHousing.csv"
})
