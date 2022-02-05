import torch

from ..registry import registry

from .modeling_utils import ValuePredictionHeadPrositFragmentation, SimpleLinear
from .modeling_bert import ProteinBertAbstractModel, ProteinBertModel, ProteinBertEncoder, ProteinBertPooler
from .modeling_lstm import ProteinLSTMEncoder, ProteinLSTMPooler, ProteinLSTMAbstractModel, ProteinLSTMModel

@registry.register_task_model('prosit_fragmentation_rnn_decoder', 'transformer')
class ProteinBertForValuePredictionFragmentationProsit(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = ProteinBertModel(config)
        #Hardcode extra dim and output for now
        self.predict = ValuePredictionHeadPrositFragmentation(config.hidden_size, 174, config.final_layer_dropout_prob, config.delta)

        self.init_weights()
        self.meta_dense = SimpleLinear(7, config.hidden_size, config.final_layer_dropout_prob, True)

        tmp_config = type('tmp_config', (object,), dict(vars(config), num_hidden_layers=10))
        tmp_config = type('tmp_config', (object,), dict(vars(tmp_config), input_size=768))

        self.layer = ProteinLSTMEncoder(tmp_config) 

        self.pooler = ProteinLSTMPooler(tmp_config)


    def forward(self, input_ids, collision_energy, charge, input_mask=None, targets=None):

        outputs = self.bert(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]

        meta_data = torch.cat((charge, collision_energy[:,None]), dim=1)

        extended_attention_mask = input_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        meta = self.meta_dense(meta_data)

        x = meta[:,None,:] * sequence_output

        x = self.layer(x)

        

        pooled_output = self.pooler(x[1])

        
        outputs = self.predict(pooled_output, targets) + outputs[2:]

        return outputs

@registry.register_task_model('prosit_fragmentation_rnn_encoder', 'lstm')
class ProteinBertForValuePredictionFragmentationProsit(ProteinLSTMAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.lstm = ProteinLSTMModel(config)
        #Hardcode extra dim and output for now
        

        self.init_weights()
        self.meta_dense = SimpleLinear(7, config.hidden_size * 2, config.final_layer_dropout_prob, True)

        tmp_config = type('tmp_config', (object,), dict(vars(config), hidden_size=config.hidden_size * 2))
        #tmp_config = type('tmp_config', (object,), dict(vars(config), input_size=1536))
        #tmp_config = type('tmp_config', (object,), dict(vars(config), input_size=1536))
        
        self.predict = ValuePredictionHeadPrositFragmentation(tmp_config.hidden_size, 174, tmp_config.final_layer_dropout_prob, tmp_config.delta)

        

        self.layer = ProteinBertEncoder(tmp_config) 

        self.pooler = ProteinBertPooler(tmp_config)


    def forward(self, input_ids, collision_energy, charge, input_mask=None, targets=None):

        outputs = self.lstm(input_ids, input_mask=input_mask)

        sequence_output, pooled_output = outputs[:2]

        meta_data = torch.cat((charge, collision_energy[:,None]), dim=1)

        meta = self.meta_dense(meta_data)

        x = meta[:,None,:] * sequence_output

        
        extended_attention_mask = input_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        

        x = self.layer(x, extended_attention_mask)[0]
        
        pooled_output = self.pooler(x)

        


        outputs = self.predict(pooled_output, targets) + outputs[2:]

        return outputs