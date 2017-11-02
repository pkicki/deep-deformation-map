from data_providers.DataProvider import Producer
from models.EncoderDecoder import Normal


def init_data_provider(_):
    data_provider = Producer(path="data.txt", batch_size=64)
    data_provider.load()
    return data_provider

# def init_model(_):
model = Normal()
model.get()



