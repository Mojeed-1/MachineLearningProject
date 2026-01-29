import sagemaker
from sagemaker.sklearn.model import SKLearnModel

MODEL_ARTIFACT = "s3://projectcontinue/path-to-model/model.tar.gz"


def deploy_model():

    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()

    model = SKLearnModel(
        model_data=MODEL_ARTIFACT,
        role=role,
        entry_point="inference.py",
        source_dir=".",
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=sagemaker_session,
    )

    predictor = model.deploy(
        instance_type="ml.m5.large",
        initial_instance_count=1,
        endpoint_name="iris-endpoint",
    )

    return predictor


if __name__ == "__main__":
    deploy_model()
