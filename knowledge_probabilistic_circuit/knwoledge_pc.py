import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import math

class knowledge_pc(nn.Module):
    def __init__(self, label_sensor_list, attr_sensor_list, knowledge, knowledge_weight, args=None, train=False):
        super(knowledge_pc, self).__init__()

        self.label_sensor_list = label_sensor_list
        self.attr_sensor_list = attr_sensor_list

        # knowledge: dim of each element: MxN
        # knowledge weight: dim of each element: MxN
        self.knowledge = knowledge

        num_label = len(label_sensor_list)
        num_attribute = len(attr_sensor_list)
        if train:
            # knowledge_weight = torch.tensor(knowledge_weight,dtype=torch.float).cuda()
            knowledge_weight = torch.rand((num_label, num_attribute),requires_grad=True).cuda()
            self.knowledge_weight = nn.Parameter(knowledge_weight)
        else:
            self.knowledge_weight = knowledge_weight
        self.num_label_sensor = len(label_sensor_list)
        self.num_attr_sensor = len(attr_sensor_list)

        if args.dataset=='GTSRB':
            # GTSRB
            self.knowledge_shape = [[[0], [0]], [[1], [1]], [[2, 3], [2]], [[4, 5, 6, 7, 8, 9, 10, 11], [3]]]
            self.knowledge_content = [[[2, 7], [4]], [[8], [8]], [[10], [9]], [[11], [10]], [[4], [11]], [[5], [12]]]
            self.knowledge_color = [[[0, 2, 3, 6, 7, 8, 10, 11], [5]], [[4, 5], [6]], [[9], [7]]]
            self.knowledge_pc = [self.knowledge_shape, self.knowledge_content, self.knowledge_color]
        elif args.dataset=='cifar10':
            self.knowledge_category = [[[2,3,4,5,6,7],[0]],[[0,1,8,9],[1]]]
            self.knowledge_place = [[[0,2],[2]],[[8],[3]],[[1,3,4,5,6,7,9],[4]]]
            self.knowledge_wheel_leg = [[[3,4,5,6,7],[5]],[[2],[6]],[[0,1,9],[7]],[[8],[8]]]
            self.knowledge_pc = [self.knowledge_category, self.knowledge_place, self.knowledge_wheel_leg]
        # self.complement_knowledge()
        self.label_count = None
        self.attr_count = None
        self.knowledge_count()
        self.fac_tables = []
        self.z2 = []
        self.z1 = []

        self.args = args
        # # default: self.weights_beta = [1.0, 1.0, 1.0]
        # self.weights_beta = [2.4, 0.3, 0.3]

    def knowledge_count(self):
        self.label_count = torch.zeros((self.num_label_sensor)).cuda()
        self.attr_count = torch.zeros((self.num_attr_sensor)).cuda()
        for knowledge in self.knowledge_pc:
            for part in knowledge:
                for tmp in part[0]:
                    self.label_count[tmp] += 1.0
                for tmp in part[1]:
                    self.attr_count[tmp] += 1.0

    # def complement_knowledge(self):
    #     for knowledge in self.knowledge_pc:
    #         label_list = list(range(self.num_label_sensor))
    #         attr_list = list(range(self.num_attr_sensor))
    #         for part in knowledge:
    #             part0 = part[0]
    #             part1 = part[1]
    #             for tmp in part0:
    #                 label_list.remove(tmp)
    #             for tmp in part1:
    #                 attr_list.remove(tmp)
    #         knowledge.append([label_list, attr_list])

    def forward(self, x):
        res = []
        for i in range(self.num_label_sensor):
            res_i = self.label_sensor_list[i](x)
            res.append(res_i[:,:].sigmoid())
        res = torch.stack(res,dim=1)

        pred_attr = []
        for i in range(self.num_attr_sensor):
            pred_attr_i = self.attr_sensor_list[i](x)
            pred_attr.append(pred_attr_i[:,:].sigmoid())
        pred_attr = torch.stack(pred_attr,dim=1)
        if self.args.pc_correction:
            res = self.knowledge_correction(res,pred_attr)
        else:
            res = torch.concat([res, pred_attr], dim=1)
        return res

    def _count_arr(self, arr, length):
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def smoothed_forward(self, model, x, sigma=0.5, N=100000, batch_size=5000, pc_correction=True):
        with torch.no_grad():
            counts = np.zeros(self.num_label_sensor+self.num_attr_sensor, dtype=int)
            for _ in range(math.ceil(N / batch_size)):
                this_batch_size = min(batch_size, N)
                N -= this_batch_size
                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * sigma
                predictions = model(batch + noise)[:,:].argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_label_sensor+self.num_attr_sensor)
        return counts

    def smoothed_forward_2(self, x, sigma=0.5, N=100000, batch_size=5000, pc_correction=True):
        with torch.no_grad():
            N_ = N
            prob = torch.zeros((self.num_label_sensor+self.num_attr_sensor,2)).cuda()
            self.args.pc_correction = 0
            for _ in range(math.ceil(N / batch_size)):
                this_batch_size = min(batch_size, N)
                N -= this_batch_size
                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * sigma
                predictions = self.forward(batch + noise)
                prob += predictions.mean(0) / math.ceil(N_ / batch_size)
            self.args.pc_correction = 1
        return prob

    def certify_PC(self, pred_label_lowbnd, pred_label_upbnd, pred_attr_lowbnd, pred_attr_upbnd, pc_correction=True):
        knowledge_pc = self.knowledge_pc
        pred_label_lowbnd = torch.from_numpy(pred_label_lowbnd).cuda()
        pred_label_upbnd = torch.from_numpy(pred_label_upbnd).cuda()
        pred_attr_lowbnd = torch.from_numpy(pred_attr_lowbnd).cuda()
        pred_attr_upbnd = torch.from_numpy(pred_attr_upbnd).cuda()
        pred_label_correct_lowbnd = torch.zeros_like(pred_label_lowbnd).cuda()
        pred_label_correct_upbnd = torch.zeros_like(pred_label_upbnd).cuda()
        pred_attr_correct_lowbnd = torch.zeros_like(pred_attr_lowbnd).cuda()
        pred_attr_correct_upbnd = torch.zeros_like(pred_attr_upbnd).cuda()

        if pc_correction==False:
            sensor_lowbnd_correct = torch.concat([pred_label_lowbnd, pred_attr_lowbnd], dim=1)
            sensor_upbnd_correct = torch.concat([pred_label_upbnd, pred_attr_upbnd], dim=1)
            return sensor_lowbnd_correct, sensor_upbnd_correct

        # print(f'pred_label_lowbnd: {pred_label_lowbnd}')
        # print(f'pred_label_upbnd: {pred_label_upbnd}')
        # print(f'pred_attr_lowbnd: {pred_attr_lowbnd}')
        # print(f'pred_attr_upbnd: {pred_attr_upbnd}')

        for i in range(len(knowledge_pc)):
            knowledge = knowledge_pc[i]
            for part in knowledge:
                num_label = len(part[0])
                num_attr = len(part[1])
                num = num_label + num_attr

                # certification for label (la=0) and attribute (la=1)
                for la in range(2):
                    for id_kk, kk in enumerate(part[la]):
                        z_12_min_max = []
                        for idx in range(4):
                            # idx=0->z1_min; idx=1->z1_max; idx=2->z2_min; idx=3->z2_max
                            z = torch.zeros(pred_label_lowbnd.shape[0]).cuda()
                            factor_table = torch.zeros((pred_label_lowbnd.shape[0], int(2 ** num))).cuda()
                            for j in range(int(2 ** num)):
                                assign_label, assign_attr = self.int2vec(j, num_label, num_attr)
                                for k in range(len(assign_label)):
                                    if k==id_kk and la==0:
                                        factor_table[:, j] += 0.0
                                    elif assign_label[k] == 1:
                                        if idx == 0 or idx == 2:
                                            tmp = torch.log(pred_label_upbnd[:, part[0][k]])
                                        elif idx == 1 or idx == 3:
                                            tmp = torch.log(pred_label_lowbnd[:, part[0][k]])
                                        # if (idx<2 and la==0 and assign_label[id_kk]==0) or (idx<2 and la==1 and assign_attr[id_kk]==0) \
                                        #         or (idx>=2 and la==0 and assign_label[id_kk]==1) or (idx>=2 and la==1 and assign_attr[id_kk]==1):
                                        factor_table[:, j] += tmp
                                    elif assign_label[k] == 0:
                                        if idx == 0 or idx == 2:
                                            tmp = torch.log(1-pred_label_upbnd[:, part[0][k]])
                                        elif idx == 1 or idx == 3:
                                            tmp = torch.log(1-pred_label_lowbnd[:, part[0][k]])
                                        # if (idx < 2 and la == 0 and assign_label[id_kk] == 0) or (
                                        #         idx < 2 and la == 1 and assign_attr[id_kk] == 0) \
                                        #         or (idx >= 2 and la == 0 and assign_label[id_kk] == 1) or (
                                        #         idx >= 2 and la == 1 and assign_attr[id_kk] == 1):
                                        factor_table[:, j] += tmp
                                for k in range(len(assign_attr)):
                                    if k==id_kk and la==1:
                                        factor_table[:, j] += 0.0
                                    elif assign_attr[k] == 1:
                                        if idx == 0 or idx == 2:
                                            tmp = torch.log(pred_attr_lowbnd[:, part[1][k]])
                                        elif idx == 1 or idx == 3:
                                            tmp = torch.log(pred_attr_upbnd[:, part[1][k]])
                                        # if (idx < 2 and la == 0 and assign_label[id_kk] == 0) or (
                                        #         idx < 2 and la == 1 and assign_attr[id_kk] == 0) \
                                        #         or (idx >= 2 and la == 0 and assign_label[id_kk] == 1) or (
                                        #         idx >= 2 and la == 1 and assign_attr[id_kk] == 1):
                                        factor_table[:, j] += tmp
                                    elif assign_attr[k] == 0:
                                        if idx == 0 or idx == 2:
                                            tmp = torch.log(1-pred_attr_lowbnd[:, part[1][k]])
                                        elif idx == 1 or idx == 3:
                                            tmp = torch.log(1-pred_attr_upbnd[:, part[1][k]])
                                        # if (idx < 2 and la == 0 and assign_label[id_kk] == 0) or (
                                        #         idx < 2 and la == 1 and assign_attr[id_kk] == 0) \
                                        #         or (idx >= 2 and la == 0 and assign_label[id_kk] == 1) or (
                                        #         idx >= 2 and la == 1 and assign_attr[id_kk] == 1):
                                        factor_table[:, j] += tmp
                                for k1 in range(len(assign_label)):
                                    for k2 in range(len(assign_attr)):
                                        if self.knowledge[part[0][k1]][part[1][k2]] == 1 and (
                                                assign_label[k1] == 0 or (assign_label[k1] == 1 and assign_attr[k2] == 1)):
                                            factor_table[:, j] += self.knowledge_weight[part[0][k1]][part[1][k2]]
                                        # TODO: consider knowledge=0? also useful knowledge (must be encoded in another PC)
                                factor_table[:, j] = torch.exp(factor_table[:, j])

                            # if i==0 and part[0][0]==2 and (la==0 and id_kk==1) and idx==3:
                            #     for j in range(int(2 ** num)):
                            #         assign_label, assign_attr = self.int2vec(j, num_label, num_attr)
                            #         for k1 in range(len(assign_label)):
                            #             for k2 in range(len(assign_attr)):
                            #                 print(self.knowledge[part[0][k1]][part[1][k2]])
                            #                 print(assign_label[k1])
                            #                 print(assign_attr[k2])
                            #                 print(self.knowledge_weight[part[0][k1]][part[1][k2]])
                            #     print('factor table')
                            #     print(factor_table)

                            for j in range(int(2 ** num)):
                                assign_label, assign_attr = self.int2vec(j, num_label, num_attr)
                                if la==0:
                                    if assign_label[id_kk]==0 and idx<2:
                                        z += factor_table[:,j]
                                    elif assign_label[id_kk]==1 and idx>=2:
                                        z += factor_table[:,j]
                                elif la==1:
                                    if assign_attr[id_kk]==0 and idx<2:
                                        z += factor_table[:,j]
                                    elif assign_attr[id_kk]==1 and idx>=2:
                                        z += factor_table[:,j]
                            if la==0:
                                if idx==0:
                                    z *= (1 - pred_label_upbnd[:,kk])
                                elif idx==1:
                                    z *= (1 - pred_label_lowbnd[:,kk])
                                elif idx==2:
                                    z *= pred_label_lowbnd[:,kk]
                                elif idx==3:
                                    z *= pred_label_upbnd[:,kk]
                            elif la==1:
                                if idx==0:
                                    z *= (1 - pred_attr_upbnd[:,kk])
                                elif idx==1:
                                    z *= (1 - pred_attr_lowbnd[:,kk])
                                elif idx==2:
                                    z *= pred_attr_lowbnd[:,kk]
                                elif idx==3:
                                    z *= pred_attr_upbnd[:,kk]
                            z_12_min_max.append(z)
                        if la==0:
                            pred_label_correct_lowbnd[:,kk] += (z_12_min_max[1] / z_12_min_max[2] + 1.0)**(-1)
                            pred_label_correct_upbnd[:,kk] += (z_12_min_max[0] / z_12_min_max[3] + 1.0)**(-1)
                        elif la==1:
                            pred_attr_correct_lowbnd[:, kk] += (z_12_min_max[1] / z_12_min_max[2] + 1.0) ** (-1)
                            pred_attr_correct_upbnd[:, kk] += (z_12_min_max[0] / z_12_min_max[3] + 1.0) ** (-1)
        pred_label_correct_lowbnd /= self.label_count
        pred_label_correct_upbnd /= self.label_count
        pred_attr_correct_lowbnd /= self.attr_count
        pred_attr_correct_upbnd /= self.attr_count

        pred_label_correct_lowbnd = pred_label_lowbnd * (1 - self.args.pc_weight) + pred_label_correct_lowbnd * self.args.pc_weight
        pred_label_correct_upbnd = pred_label_upbnd * (1 - self.args.pc_weight) + pred_label_correct_upbnd * self.args.pc_weight
        pred_attr_correct_lowbnd = pred_attr_lowbnd * (1 - self.args.pc_weight) + pred_attr_correct_lowbnd * self.args.pc_weight
        pred_attr_correct_upbnd = pred_attr_upbnd * (1 - self.args.pc_weight) + pred_attr_correct_upbnd * self.args.pc_weight

        sensor_lowbnd_correct = torch.concat([pred_label_correct_lowbnd,pred_attr_correct_lowbnd],dim=1)
        sensor_upbnd_correct = torch.concat([pred_label_correct_upbnd, pred_attr_correct_upbnd], dim=1)

        # sensor_lowbnd_correct[sensor_lowbnd_correct<0.05] = 0.05
        # sensor_upbnd_correct[sensor_upbnd_correct>0.95] = 0.95

        return sensor_lowbnd_correct, sensor_upbnd_correct


    def knowledge_correction(self, pred_label, pred_attr):
        with torch.enable_grad():
            knowledge_pc = self.knowledge_pc


            pred_label_correct = torch.zeros_like(pred_label).cuda()
            pred_attr_correct = torch.zeros_like(pred_attr).cuda()
            self.fac_tables = []
            self.z2 = []
            self.z1 = []

            pred_label[pred_label < 1e-5] = 1e-5
            pred_attr[pred_attr < 1e-5] = 1e-5

            result_label = [{},{},{}]
            result_attr = [{},{},{}]
            for i in range(len(knowledge_pc)):
                knowledge = knowledge_pc[i]
                self.z1.append(torch.zeros((pred_label.shape[0],pred_label.shape[1]+pred_attr.shape[1])).cuda())
                ind_z1 = len(self.z1)-1

                for part in knowledge:
                    num_label = len(part[0])
                    num_attr = len(part[1])
                    num = num_label + num_attr
                    factor_table = torch.zeros((pred_label.shape[0],int(2**num))).cuda()

                    self.fac_tables.append(factor_table)
                    ind = len(self.fac_tables)-1
                    self.z2.append(torch.zeros((pred_label.shape[0])).cuda())
                    ind_z2 = len(self.z2)-1
                    # construct the factor table
                    for j in range(int(2**num)):
                        assign_label,assign_attr = self.int2vec(j,num_label,num_attr)
                        for k in range(len(assign_label)):
                            if assign_label[k]==1:
                                self.fac_tables[ind][:, j] = self.fac_tables[ind][:, j] + (pred_label[:, part[0][k], 1]).log()
                            else:
                                self.fac_tables[ind][:, j] = self.fac_tables[ind][:, j] + (pred_label[:, part[0][k], 0]).log()
                        for k in range(len(assign_attr)):
                            if assign_attr[k] == 1:
                                self.fac_tables[ind][:, j] = self.fac_tables[ind][:, j] + (pred_attr[:, part[1][k], 1]).log()
                            else:
                                self.fac_tables[ind][:, j] = self.fac_tables[ind][:, j] + (pred_attr[:, part[1][k], 0]).log()
                        for k1 in range(len(assign_label)):
                            for k2 in range(len(assign_attr)):
                                if self.knowledge[part[0][k1]][part[1][k2]]==1 and (assign_label[k1]==0 or (assign_label[k1]==1 and assign_attr[k2]==1)):
                                    self.fac_tables[ind][:, j] = (self.fac_tables[ind][:, j] + (self.knowledge_weight[part[0][k1]][part[1][k2]]))
                    self.fac_tables[ind] = torch.exp(self.fac_tables[ind])
                    self.z2[ind_z2] = torch.sum(self.fac_tables[ind],dim=1)

                    for j in range(int(2**num)):
                        assign_label,assign_attr = self.int2vec(j,num_label,num_attr)
                        for k in range(len(assign_label)):
                            if assign_label[k]==1:
                                self.z1[ind_z1][:,part[0][k]] = self.z1[ind_z1][:,part[0][k]] + self.fac_tables[ind][:,j]
                        for k in range(len(assign_attr)):
                            if assign_attr[k] == 1:
                                self.z1[ind_z1][:,num_label+part[1][k]] = self.z1[ind_z1][:,num_label+part[1][k]] + self.fac_tables[ind][:,j]
                    for k in part[0]:
                        # z2[:,k] = z2_tmp
                        # x = (pred_label_correct[:, k, 1] + self.z1[:,k] / z2_tmp)
                        # pred_label_correct[:, k, 1] = pred_label_correct[:, k, 1] + self.z1[:,k] / z2_tmp
                        # if k in result_label.keys():
                        #     result_label[k] = result_label[k] + self.z1[:,k] / z2_tmp
                        # else:
                        # self.z1[ind_z1][:, k] = torch.ones_like(self.z1[ind_z1][:, k])
                        # self.z2[ind_z2] = torch.ones_like(self.z2[ind_z2])
                        result_label[i][k] = self.z1[ind_z1][:, k] / self.z2[ind_z2]
                    for k in part[1]:
                        # z2[:,pred_label.shape[1]+k] = z2_tmp
                        # x = (pred_attr_correct[:, k, 1] + self.z1[:, self.num_label_sensor+k] / z2_tmp)
                        # pred_attr_correct[:, k, 1] = pred_attr_correct[:, k, 1] + self.z1[:, self.num_label_sensor+k] / z2_tmp
                        # result_attr[k] = self.z1[:, self.num_label_sensor+k] / z2_tmp
                        # if k in result_attr.keys():
                        #     result_attr[k] = result_attr[k] + self.z1[:, self.num_label_sensor+k] / z2_tmp
                        # else:
                        if k in result_attr[i].keys():
                            print(f'error: {k}  {i}')
                        # print(self.z1[:, self.num_label_sensor+k])
                        # self.z1[ind_z1][:, self.num_label_sensor + k] = torch.ones_like(self.z1[ind_z1][:, self.num_label_sensor+k])
                        result_attr[i][k] = self.z1[ind_z1][:, self.num_label_sensor+k] / self.z2[ind_z2]


                # pred_label_correct[:,:,1] += z1[:,:pred_label.shape[1]] / z2[:,:pred_label.shape[1]]
                # pred_attr_correct[:,:,1] += z1[:,pred_label.shape[1]:] / z2[:,pred_label.shape[1]:]
                # pred_label_correct[:, :, 0] += 1-pred_label_correct[:, :, 1]
                # pred_attr_correct[:, :, 0] += 1-pred_attr_correct[:,:,1]


            for k in result_label[0].keys():
                pred_label_correct[:,k,1] = result_label[0][k] / self.label_count[k]
            for k in result_attr[0].keys():
                pred_attr_correct[:,k,1] = result_attr[0][k] / self.attr_count[k]

            # pred_label_correct[:,:,1] = pred_label_correct[:,:,1] / self.label_count
            # pred_attr_correct[:,:,1] = pred_attr_correct[:,:,1] / self.attr_count
            pred_label_correct[:, :, 0] = 1-pred_label_correct[:, :, 1]
            pred_attr_correct[:, :, 0] = 1-pred_attr_correct[:, :, 1]
            # pred_correct = torch.concat([pred_label_correct, pred_attr_correct], dim=1)

            # take the average
            pred_label_correct = pred_label * (1-self.args.pc_weight) + pred_label_correct * self.args.pc_weight
            pred_attr_correct = pred_attr * (1-self.args.pc_weight) + pred_attr_correct * self.args.pc_weight

            pred_correct = torch.concat([pred_label_correct,pred_attr_correct],dim=1)
            # pred_correct[pred_correct > 0.999] = 0.999
            # pred_correct[pred_correct < 0.001] = 0.001
            return pred_correct


    def int2vec(self,j,num_label,num_attr):
        num = num_label + num_attr
        res = [0] * num
        for i in reversed(range(num)):
            res[i] = j % 2
            j = j // 2
        return res[:num_label], res[num_label:]

    def knowledge_correction_full(self, pred_label, pred_attr):

        self.construct_factor_table(pred_label, pred_attr)

        print('print some factor value in the table')
        for i in range(20):
            print(f'factor value {i}: {self.factor_table[i]}')

        num = self.num_label_sensor + self.num_attr_sensor
        pred_correct = torch.zeros((pred_label.shape[0],num)).cuda()

        z2 = torch.zeros((pred_attr.shape[0])).cuda()
        for k in tqdm(range(self.factor_table.shape[1])):
            z2 += torch.exp(self.factor_table[:,k].cuda())
        print(f'z2: {z2}')

        print('inference label')
        for i in tqdm(range(pred_label.shape[1])):
            z1 = torch.zeros((pred_label.shape[0])).cuda()
            for k in range(self.factor_table.shape[1]):
                if k>>(num-1-i)%2==1:
                    z1 += torch.exp(self.factor_table[:,k].cuda())
            pred_correct[:,i] = z1 / z2

        print('inference attribute')
        for j in range(pred_attr.shape[1]):
            z1 = torch.zeros((pred_attr.shape[0])).cuda()
            for k in range(self.factor_table.shape[1]):
                if k>>(num-1-j-self.num_label_sensor)%2==1:
                    z1 += torch.exp(self.factor_table[:,k].cuda())
            pred_correct[:, j+self.num_label_sensor] = z1 / z2

        print('pred_correct')
        print(pred_correct[0])

        return pred_correct

    def vec2int(self,x):
        base = 1
        res = 0
        for i in reversed(list(range(len(x)))):
            res += base * x[i]
            base *= 2
        return res

    def construct_factor_table(self, pred, pred_attr):
        num = self.num_label_sensor + self.num_attr_sensor
        self.factor_table = torch.zeros(pred.shape[0],int(2**num))

        print('construct label->attribute knowledge')
        for i in tqdm(range(self.knowledge.shape[0])):
            for j in range(self.knowledge.shape[1]):
                if self.knowledge[i][j]==1:
                    for k in range(self.factor_table.shape[1]):
                        if (k>>(num-1-i))%2==1 and (k>>(num-1-j-self.num_label_sensor))%2==1:
                            self.factor_table[:,k] += self.knowledge_weight[i][j].detach().cpu()

        print('construct label knowledge')
        for i in tqdm(range(pred.shape[1])):
            for k in range(self.factor_table.shape[1]):
                if (k>>(num-1-i))%2==1:
                    self.factor_table[:,k] += torch.log(pred[:,i,1]/(1-pred[:,i,1])).detach().cpu()
                else:
                    self.factor_table[:, k] += torch.log(pred[:, i, 0] / (1 - pred[:, i, 0])).detach().cpu()

        print('construct attribute knowledge')
        for i in tqdm(range(pred_attr.shape[1])):
            for k in range(self.factor_table.shape[1]):
                if k>>(num-1-j-self.num_label_sensor)%2==1:
                    self.factor_table[:, k] += torch.log(pred_attr[:, i,1] / (1 - pred_attr[:, i,1])).detach().cpu()
                else:
                    self.factor_table[:, k] += torch.log(pred[:, i, 0] / (1 - pred[:, i, 0])).detach().cpu()

