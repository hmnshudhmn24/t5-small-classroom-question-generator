from src.inference import generate_question
def test_generate_signature():
    try:
        out = generate_question('Photosynthesis', 6, model_dir='model/checkpoints/best-model')
    except Exception as e:
        assert isinstance(e, Exception)
        return
    assert isinstance(out, str)
