from model import train_model

def test_model_accuracy():
    accuracy = train_model()
    assert accuracy['eval_accuracy'] > 0.5, "Model accuracy is too low!"
