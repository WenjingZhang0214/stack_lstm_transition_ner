# -*- coding:utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import utils.utils as utils
import numpy as np

class TRAN_NER(nn.Module):

    def __init__(self, word_dict, action_dict, vocab_size, actions_size, embedding_dim, hidden_dim, action_dim, relation_dim, rnn_layers, dropout_ratio):
        super(TRAN_NER, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.relation_dim = relation_dim

        self.word_dict = word_dict
        self.action_dict = action_dict

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.action_embeds = nn.Embedding(actions_size, action_dim)
        self.relation_embeds = nn.Embedding(actions_size, relation_dim)#?

        self.stack_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=rnn_layers, dropout=dropout_ratio)
        self.buffer_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=rnn_layers, dropout=dropout_ratio)
        self.output_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=rnn_layers, dropout=dropout_ratio)
        self.action_lstm = nn.LSTM(action_dim, hidden_dim, num_layers=rnn_layers, dropout=dropout_ratio)
        self.entity_forward_lstm = nn.LSTM(embedding_dim, embedding_dim)
        self.entity_backward_lstm = nn.LSTM(embedding_dim, embedding_dim)

        self.rnn_layers = rnn_layers

        self.dropout1 = nn.Dropout(p=dropout_ratio)
        self.dropout2 = nn.Dropout(p=dropout_ratio)

        self.batch_size = 1
        self.seq_length = 1

        self.combine_lstms_output=nn.Linear(hidden_dim *4,hidden_dim)
        self.combinelstmsoutput2action=nn.Linear(hidden_dim,actions_size)
        self.entrela2action = nn.Linear(embedding_dim*2+relation_dim, embedding_dim)

        self.action_start = nn.Parameter(torch.randn(1,self.batch_size,action_dim))
        self.stack_start = nn.Parameter(torch.randn(1,self.batch_size,embedding_dim))
        self.output_start = nn.Parameter(torch.randn(1,self.batch_size,embedding_dim))
        self.buffer_start = nn.Parameter(torch.randn(1,self.batch_size,embedding_dim))

        self.idx2action = {v: k for k, v in action_dict.items()}#action_dict=(action,id)

        #self.m = nn.LogSoftmax()

    def set_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        tmp = sentence.size()
        self.seq_length = tmp[0]
        self.batch_size = tmp[1]

    def load_pretrained_embedding(self, pre_embeddings):
        """
        load pre-trained word embedding

        args:
            pre_word_embeddings (self.word_size, self.word_dim) : pre-trained embedding
        """
        assert (pre_embeddings.size()[1] == self.embedding_dim)
        self.word_embeds.weight = nn.Parameter(pre_embeddings)


    def rand_init(self, init_embedding=False):
        """
        random initialization

        args:
            init_embedding: random initialize embedding or not
        """
        if init_embedding:
            utils.init_embedding(self.word_embeds.weight)

        #init action_embeds relation_embeds
        utils.init_embedding(self.action_embeds.weight)
        utils.init_embedding(self.relation_embeds.weight)

        #init lstms
        utils.init_lstm(self.stack_lstm)
        utils.init_lstm(self.buffer_lstm)
        utils.init_lstm(self.output_lstm)
        utils.init_lstm(self.action_lstm)
        utils.init_lstm(self.entity_backward_lstm)
        utils.init_lstm(self.entity_forward_lstm)

        #init linear
        utils.init_linear(self.combine_lstms_output)
        utils.init_linear(self.combinelstmsoutput2action)
        utils.init_linear(self.entrela2action)

    def forward(self, sentence, actions, if_cuda, hidden=None):
        self.set_batch_seq_size(sentence)

        #d_word_embeds = self.dropout1(self.word_embeds(sentence))#seqlen,batch,embsize

        rev_sentence_list = list()
        sentence_list = sentence.squeeze(1).data.tolist()
        '''
        idx2feature = {v: k for k, v in self.word_dict.items()}
        sentence_str = list()
        for i in sentence_list:
            sentence_str.append(idx2feature[i])
        '''
        for i,e in enumerate(sentence_list):
            rev_sentence_list.append(sentence_list[len(sentence_list)-1-i])
        if if_cuda:
            rev_sentence = autograd.Variable(torch.LongTensor(rev_sentence_list).unsqueeze(1)).cuda()
        else:
            rev_sentence = autograd.Variable(torch.LongTensor(rev_sentence_list).unsqueeze(1))
        reverse_word_embeds = self.word_embeds(rev_sentence)  # seqlen,batch,embsize

        reverse_d_word_embeds = self.dropout1(reverse_word_embeds)

        buffer = list()
        stack = list()
        output = list()

        stack.append(self.stack_start)
        output.append(self.output_start)
        buffer.append(self.buffer_start)

        #x=len(sentence_array)-1
        #y=d_word_embeds[x]

        for idx in range(len(sentence_list)):
            buffer.append(reverse_d_word_embeds[idx].unsqueeze(0))#buffer是list，长度是seq_len+1,但是每个元素是1*100，也就是batch*embed,所以需要.unsqueeze(0)


        cat_word_embeds = torch.cat([self.buffer_start, reverse_d_word_embeds], 0)#seq_len+1,batch,embsize

        # output, (h_n, c_n)
        # output (seq_len, batch, hidden_size)
        # h_n (num_layers, batch, hidden_size)
        # c_n (num_layers, batch, hidden_size)
        buffer_lstm_output, buffer_lstm_hidden = self.buffer_lstm(cat_word_embeds, hidden)
        stack_lstm_output, stack_lstm_hidden = self.stack_lstm(stack[-1],hidden)
        output_lstm_output, output_lstm_hidden = self.output_lstm(output[-1],hidden)
        action_lstm_output, action_lstm_hidden = self.action_lstm(self.action_start, hidden)

        stack_lstm_hidden_list = list()
        stack_lstm_hidden_list.append(stack_lstm_hidden)

        stack_lstm_index = 0
        buffer_lstm_index = self.seq_length #初始时，index都指向最后一个output


        action_count = 0
        right = 0
        losses = list()
        results = list()
        while len(buffer)>1 or len(stack)>1:

            current_valid_actions = list()
            for a in self.action_dict:
                if utils.IsActionForbidden(a, len(buffer), len(stack)):
                    continue
                current_valid_actions.append(self.action_dict[a])
            current_valid_actions = sorted(current_valid_actions)


            lstms_output = torch.cat([output_lstm_output[-1], stack_lstm_output[-1], buffer_lstm_output[buffer_lstm_index], action_lstm_output[-1]], 1)#batch*(hidden*4)
            linear_lstms_output = torch.tanh(self.combine_lstms_output(self.dropout2(lstms_output)))#batch*hidden
            #m = nn.ReLU()
            #relu_lstms_output = m(linear_lstms_output)
            actions_output = self.combinelstmsoutput2action(linear_lstms_output)#1*6 batch,action_size

            if if_cuda:
                logits = actions_output[0][autograd.Variable(torch.LongTensor(current_valid_actions)).cuda()]
            else:
                logits = actions_output[0][autograd.Variable(torch.LongTensor(current_valid_actions))]#这个0是因为batch=1，所以只有1行
            log_probs = nn.functional.log_softmax(logits)
            log_probs_array = log_probs.data.cpu().numpy()

            max_action_score = torch.max(log_probs)
            max_action_id = current_valid_actions[log_probs_array.argmax()]
            max_action = self.idx2action[max_action_id]  # shift out reduce

            right_action_id = max_action_id
            right_action_score = max_action_score
            right_action = max_action

            current_valid_actions_dict = {a: i for i, a in enumerate(current_valid_actions)}

            if len(actions)>0:
                actions_array = actions.data.cpu().numpy()
                #得到当前正确的action的分数
                right_action_id = int(actions_array[action_count][0])
                right_action = self.idx2action[right_action_id]
                right_action_score = log_probs[current_valid_actions_dict[right_action_id]]

                if right_action_id == max_action_id:
                    right += 1

            action_count +=1

            losses.append(right_action_score)
            results.append(right_action_id)

            if if_cuda:
                tmp = autograd.Variable(torch.LongTensor([right_action_id]).unsqueeze(1), requires_grad=False).cuda()
            else:
                tmp = autograd.Variable(torch.LongTensor([right_action_id]).unsqueeze(1), requires_grad=False)
            #d_action_embeds = self.dropout1(self.action_embeds(autograd.Variable(torch.LongTensor([right_action_id]).unsqueeze(1), requires_grad=False)))
            d_action_embeds = self.dropout1(self.action_embeds(tmp))
            #h_0 = action_lstm_hidden[0]
            #c_0 = action_lstm_hidden[1]
            #action_lstm_output, action_lstm_hidden = self.action_lstm(d_action_embeds,(h_0,c_0))
            action_lstm_output, action_lstm_hidden = self.action_lstm(d_action_embeds, action_lstm_hidden)

            #cat_action_embeds = torch.cat([self.action_start, d_action_embeds], 0)
            #action_lstm_output_all, action_lstm_hidden_all = self.action_lstm(cat_action_embeds, hidden)#不一样

            if if_cuda:
                tmp = autograd.Variable(torch.LongTensor([right_action_id]).unsqueeze(1), requires_grad=False).cuda()
            else:
                tmp = autograd.Variable(torch.LongTensor([right_action_id]).unsqueeze(1), requires_grad=False)
            d_relation_embeds = self.dropout1(self.relation_embeds(tmp))#1*batch*relation_embeds

            if right_action.startswith("S"):
                assert (len(buffer) > 1)

                stack.append(buffer.pop())
                #h_0 = stack_lstm_hidden[0]
                #c_0 = stack_lstm_hidden[1]
                #stack_lstm_output_new, stack_lstm_hidden = self.stack_lstm(stack[-1], (h_0, c_0))
                stack_lstm_output_new, stack_lstm_hidden = self.stack_lstm(stack[-1], stack_lstm_hidden)
                stack_lstm_output = torch.cat([stack_lstm_output, stack_lstm_output_new], 0)
                #stack_lstm_hidden=? 是否也需要保存下来
                stack_lstm_hidden_list.append(stack_lstm_hidden)
                stack_lstm_index += 1

                buffer_lstm_index -= 1
                #buffer_lstm_hidden[0] = buffer_lstm_output[buffer_lstm_index]
            elif right_action.startswith("R"):
                cat_entity_forward = stack[0]
                cat_entity_backward = stack[-1]
                #for i in stack[1:]:
                for i in range(1,len(stack)):
                    cat_entity_forward = torch.cat([cat_entity_forward, stack[i]], 0)  # len(stack),batch,word_embeds
                    cat_entity_backward = torch.cat([cat_entity_backward, stack[len(stack)-1-i]], 0)  # len(stack),batch,word_embeds
                entity_forward_lstm_output, entity_forward_lstm_hidden = self.entity_forward_lstm(cat_entity_forward, hidden)
                entity_backward_lstm_output, entity_backward_lstm_hidden = self.entity_backward_lstm(cat_entity_backward, hidden)
                i = len(stack)
                while i>1:
                    stack_lstm_output.data = stack_lstm_output.data[0:stack_lstm_index]
                    stack_lstm_hidden_list.pop()
                    stack_lstm_hidden = stack_lstm_hidden_list[-1]
                    #stack_lstm_hidden[0] = stack_lstm_hidden_list[-1][0]
                    #stack_lstm_hidden[1] = stack_lstm_hidden_list[-1][1]

                    stack_lstm_index -= 1
                    stack.pop()
                    i -= 1
                efwd = self.dropout2(entity_forward_lstm_output.data[-1].unsqueeze(0))#1,batch,entity_hidden_size 1*1*20
                ebwd = self.dropout2(entity_backward_lstm_output.data[-1].unsqueeze(0))
                combine_entity = torch.cat([efwd, ebwd], 2)#1,batch,entity_hidden_size*2 1*1*40
                combine_entity_relation = torch.cat([combine_entity, d_relation_embeds], 2) #1,batch,entity_hidden_size*2+relation_dim
                output_in = torch.tanh(self.entrela2action(combine_entity_relation))#1,batch,embedding_size

                output.append(output_in)

                #h_0 = output_lstm_hidden[0]
                #c_0 = output_lstm_hidden[1]
                #output_lstm_output_new, output_lstm_hidden = self.output_lstm(output_in, (h_0, c_0))
                output_lstm_output_new, output_lstm_hidden = self.output_lstm(output_in, output_lstm_hidden)
                output_lstm_output = torch.cat([output_lstm_output, output_lstm_output_new], 0)
            elif right_action.startswith("O"):
                assert (len(buffer) > 1)
                output.append(buffer.pop())

                #h_0 = output_lstm_hidden[0]
                #c_0 = output_lstm_hidden[1]
                #output_lstm_output_new, output_lstm_hidden = self.output_lstm(output[-1], (h_0, c_0))
                output_lstm_output_new, output_lstm_hidden = self.output_lstm(output[-1], output_lstm_hidden)
                output_lstm_output = torch.cat([output_lstm_output, output_lstm_output_new], 0)

                buffer_lstm_index -= 1
        loss = -torch.sum(torch.cat(losses))
        return loss, results, right
