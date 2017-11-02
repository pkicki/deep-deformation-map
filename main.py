from data_providers.DataProvider import Producer

data_provider = Producer("data/data.txt", 64)
data_provider.load()

data_provider()

