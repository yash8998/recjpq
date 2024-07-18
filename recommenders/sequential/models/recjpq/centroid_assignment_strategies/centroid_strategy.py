from transformers import T5Tokenizer, T5EncoderModel


class CentroidAssignmentStragety(object):
    def __init__(self, item_code_bytes, num_items) -> None:
        self.item_code_bytes = item_code_bytes
        self.num_items = num_items
        # self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        # self.model = T5EncoderModel.from_pretrained('t5-base')

    def assign(self, train_users):
        raise NotImplementedError()