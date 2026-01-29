# pipeline.py

import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.parameters import ParameterString


def get_pipeline():

    sagemaker_session = sagemaker.session.Session()
    role = sagemaker.get_execution_role()

    train_data = ParameterString(
        name="TrainData",
        default_value="s3://projectcontinue/data/train/BostonHousing.csv"
    )

    estimator = SKLearn(
        entry_point="train.py",
        source_dir=".",
        role=role,
        instance_type="ml.m5.large",
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=sagemaker_session,
    )

    training_step = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=train_data,
                content_type="text/csv"
            )
        },
    )

    pipeline = Pipeline(
        name="IrisEndToEndPipeline",
        parameters=[train_data],
        steps=[training_step],
        sagemaker_session=sagemaker_session,
    )

    return pipeline


if __name__ == "__main__":
    pipeline = get_pipeline()
    pipeline.upsert(role_arn=sagemaker.get_execution_role())
    pipeline.start()
