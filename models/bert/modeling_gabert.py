import torch
from torch import nn
from .modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler




class GAZEBERT_Encorder(BertPreTrainedModel):
    def __init__(self, config):
        super(METRO_Encoder, self).__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.img_dim = config.img_feature_dim 

        try:
            self.use_img_layernorm = config.use_img_layernorm
        except:
            self.use_img_layernorm = None

        self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.use_img_layernorm:
            self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.img_layer_norm_eps)

        self.apply(self.init_weights)


    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, img_feats, input_ids=None, token_type_ids=None, attention_mask=None,
            position_ids=None, head_mask=None):

        batch_size = len(img_feats)
        seq_length = len(img_feats[0])
        input_ids = torch.zeros([batch_size, seq_length],dtype=torch.long).cuda()

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        position_embeddings = self.position_embeddings(position_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Project input token features to have spcified hidden size
        img_embedding_output = self.img_embedding(img_feats)

        # We empirically observe that adding an additional learnable position embedding leads to more stable training
        embeddings = position_embeddings + img_embedding_output

        if self.use_img_layernorm:
            embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        encoder_outputs = self.encoder(embeddings,
                extended_attention_mask, head_mask=head_mask)
        sequence_output = encoder_outputs[0]

        outputs = (sequence_output,)
        if self.config.output_hidden_states:
            all_hidden_states = encoder_outputs[1]
            outputs = outputs + (all_hidden_states,)
        if self.config.output_attentions:
            all_attentions = encoder_outputs[-1]
            outputs = outputs + (all_attentions,)

        return outputs



class GAZEBERT(BertPreTrainedModel):
    '''
    The archtecture of a transformer encoder block we used in GAZEBERT
    '''
    def __init__(self, config):
        super(GAZEBERT, self).__init__(config)
        self.config = config
        self.bert = GAZEBERT_Encorder(config)
        self.cls_head = nn.Linear(config.hidden_size, self.config.output_feature_dim)
        self.residual = nn.Linear(config.img_feature_dim, self.config.output_feature_dim)
        self.apply(self.init_weights)

    def forward(self, img_feats, input_ids=None, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
            next_sentence_label=None, position_ids=None, head_mask=None):
        '''
        # self.bert has three outputs
        # predictions[0]: output tokens
        # predictions[1]: all_hidden_states, if enable "self.config.output_hidden_states"
        # predictions[2]: attentions, if enable "self.config.output_attentions"
        '''
        predictions = self.bert(img_feats=img_feats, input_ids=input_ids, position_ids=position_ids,
                            token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=head_mask)

        pred_score = self.cls_head(predictions[0])
        res_img_feats = self.residual(img_feats)
        pred_score = pred_score + res_img_feats

        if self.config.output_attentions and self.config.output_hidden_states:
            return pred_score, predictions[1], predictions[-1]
        else:
            return pred_score

class GAZEBERT_Network(torch.nn.Module):

    def __init__(self, args, config, backbone, trans_encoder):
        super(GAZEBERT_Network, self).__init__()
        self.config = config
        self.config.device = args.device
        self.backbone = backbone
        #self.conv_learn_tokens = torch.nn.Conv1d(48,431+14,1)
        self.trans_encoder = trans_encoder
        self.conv_learn_tokens = torch.nn.Conv1d(48, 1)

        self.mlp_layer1 = torch.nn.Linear(48, 1)
        self.mlp_layer2 = torch.nn.Linear(2048, 256)
        self.mlp_layer3 = torch.nn.Linear(256, 3)

        self.mlp1 = torch.nn.Linear(3,256)
        self.mlp2 = torch.nn.Linear(256,3)


    def forward(self, images,test, meta_masks=None, is_train=False):
        batch_size = images.size(0)
        #print("batch size", batch_size)

        ref_gaze = [0,0,1]

        # extract image feature maps using a CNN backbone
        image_feat = self.backbone(images) # [32, 2048, 8 ,6]
        image_feat_newview  = image_feat.view(batch_size, 2048, -1)
        image_feat_newview2 = image_feat_newview.transpose(1,2) # [32, 48, 2048]
        img_tokens = self.conv_learn_tokens(image_feat_newview2) # [32,1,2048]
        #print("size of image_ feat",image_feat_newview.size())
        features = torch.cat([ref_gaze, img_tokens], dim=2) # [32, 1, 2051]

        if is_train==True:
            constant_tensor = torch.ones_like(features).cuda(self.config_device)*0.01
            features = features*meta_masks + constant_tensor*(1-meta_masks)

        # forward pass
        if self.config.output_attentions==True:
            features, all_hidden_states, att = self.trans_encoder(features)
        else:
            features = self.trans_encoder(features)

        '''
        x = self.mlp_layer1(image_feat_newview)
        x = x.transpose(1,2)
        x = self.mlp_layer2(x)
        x = self.mlp_layer3(x)
        x = x.squeeze()
        #print("size of x", x.size())
        #print("pred_gaze:",torch.Tensor(pred_gaze).size())
        
        #x = self.mlp1(test)
        #x = self.mlp2(x)
        '''
        
        return features

