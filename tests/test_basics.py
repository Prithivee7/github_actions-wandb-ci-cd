import wandb
import os
import joblib
import numpy as np
import shutil

def test_check_pred():
    with wandb.init(project="titanic-project") as run:

        artifact = run.use_artifact(
            'prithiveer/model-registry/My first Registered Model:v2', type='model')
        artifact_dir = artifact.download()
        model = joblib.load(os.path.join(artifact_dir, 'my_first_model.pkl'))
        val = model.predict(np.array([3, 1, 34, 0, 0, 8, 2]).reshape(1, -1))
        assert val == [0]
        print(val)
        print("#####################################################3")
        val = model.predict(np.array([3, 0, 61, 0, 0, 9, 0]).reshape(1, -1))
        print(val)
        assert val == [0]

        shutil.rmtree("artifacts")
        run.finish()
