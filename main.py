from data_providers.DataProvider import Producer
from models.EncoderDecoder import Normal
from trainers.Trainer import MbTrainer
from testers.Tester import MBTester
from data_providers.ImagesProvider import SequenceImagesProvider


# data
def init_data_provider():
    data_provider = Producer(path="data.txt",
                             batch_size=10)

    data_provider.load()
    return data_provider
data_pro = init_data_provider()


def init_images_provider():
    images_provider = SequenceImagesProvider(sequence_path="output.avi")
    return images_provider
data_img = init_images_provider()

# # train
# trainer = MbTrainer(model=Normal(),
#                     data_provider=data_pro,
#                     output_net_name="trained_net.meta",
#                     logs_dir="./logs")
# trainer.train(1000, 10000)

# test
tester = MBTester(data_provider=data_img,
                  net_name="trained_net.meta.meta",
                  test_method="file",
                  output_path="../data/")
tester.test()
