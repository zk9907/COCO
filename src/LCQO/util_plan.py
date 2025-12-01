import numpy as np
from collections import deque
import re
from LCQO.Constant import *
def bfs(N, pc_dict):
    distance_matrix = np.full((N, N), -1e9)
    for start_node in range(N):
        queue = deque([(start_node, 0)])  # node, distance
        while queue:
            node, distance = queue.popleft()
            for end_node in pc_dict[node]:
                if distance + 1 < MAXDISTANCE:
                    distance_matrix[start_node][end_node] = distance + 1 # 1 / (distance + 2)
                    queue.append((end_node, distance + 1))
        distance_matrix[start_node][start_node] = 0
    return distance_matrix

def pad_heights(x, padlen):
    x = x # + 1  # pad id = 0
    xlen = x.shape[0]
    if xlen < padlen:
        new_x = np.zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x #.unsqueeze(0)

def pad_1d_values(x, padlen):
    xlen = x.shape[0]
    if xlen < padlen:
        new_x = np.zeros([padlen], dtype=x.dtype) + 1e-8
        new_x[:xlen] = x
        x = new_x
    return x

def pad_2d_unsqueeze(x, padlen):
    xlen, xdim = x.shape
    # x = x + 1 # pad id = 0
    if xlen <= padlen:
        new_x = np.zeros([padlen, xdim], dtype=x.dtype) # + 1
        new_x[:xlen, :] = x
        x = new_x
    else:
        pass
        # raise ValueError(f'pad_2d_unsqueeze: xlen = {xlen}, padlen = {padlen}')
    return x

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.shape[0]
    if xlen < padlen:
        new_x = np.full([padlen, padlen], -1e9) #np.zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        # new_x[xlen:, :xlen] = 
        x = new_x
    return x

def process_condition(filt, alias, alias2table, metaInfo, filterFeature,joinFeature):
    hasFilter = False
    #  dtype: 0:number  1:text  2:NULL
    try:
        if 'IS NOT NULL' in filt:
            col = alias2table[alias] + '.' +filt.split(' ')[0].strip('()"')
            op = 'IS NOT NULL'
            dtype = 2
            val = [None]
        elif 'IS NULL' in filt:
            col = alias2table[alias] + '.' +filt.split(' ')[0].strip('()"')
            op = 'IS NULL'
            dtype = 2
            val = [None]
        elif "::" in filt:     
            filt_split = re.findall(r"(?:[^\s']+|'[^']*')+", filt)
            for i,f in enumerate(filt_split):
                if i != 0:
                    filt_split[i] = re.sub(r'::.*$', "", f)
                else:
                    match = re.search(r'\((.*?)\)::', f)
                    filt_split[i] = match.group(1) if match else re.sub(r'::.*$', "", f)
            left_split = filt_split[0].split('.')
            if len(left_split) == 1:
                left_col = alias2table[alias] + '.' + filt_split[0].strip("'()\"")
            else:
                left_col = alias2table[left_split[0].strip('"')] + '.' + left_split[1].strip("'()\"")

            right_split = [f.strip('()"') for f in filt_split[-1].split('.')]
            if len(right_split) == 1:
                right_col = alias2table[alias] + '.' + right_split[0]
            else:
                if right_split[0] in alias2table:
                    right_col = alias2table[right_split[0]] + '.' + right_split[1]
                else:
                    right_col = right_split[0] + '.' + right_split[1]
            if right_col in metaInfo:
                joinFeature.append([left_col, right_col])
                hasFilter = False

            else:
                col = left_col
                if ' = ANY ' in filt or ' <> ALL ' in filt:
                    val = filt_split[3].strip('()')
                    val = val[1:-1].strip('}{')
                    val = val.split(',')
                    op = ''.join(filt_split[1:3])
                else:
                    val = [filt_split[2].strip('()').strip("'")]
                    op = ''.join(filt_split[1])
                dtype = 1
        else:
            filt_split = filt.split(' ')
            rightColSplit = [f.strip('()"') for f in filt_split[-1].split('.')]
            if rightColSplit[0] in alias2table and alias2table[rightColSplit[0]] + '.' + rightColSplit[-1] in metaInfo:
                onejoin = []
                twoCol = [filt_split[0], filt_split[-1]]
                for col in twoCol:
                    col_split = col.split('.')
                    column = col_split[0].strip('"')
                    if len(col_split) == 1:
                        onejoin.append(alias2table[alias] + '.' + column)
                    else:
                        onejoin.append(alias2table[column] + '.' + col_split[1].strip('"'))
                joinFeature.append(onejoin)
                hasFilter = False
            else:
                col_split = [f.strip('()"') for f in filt_split[0].split('.')]
                if len(col_split) == 2:
                    col = alias2table[col_split[0]] + '.' + col_split[1]
                else:
                    col = alias2table[alias] + '.' + col_split[0]
                val = [filt_split[-1].strip('()')]
                op = ''.join(filt_split[1:-1])
                dtype = 0
            # elif alias2table[alias] + '.' + filt_split[-1].strip('()"') in metaInfo:
            #     right = alias2table[alias] + '.' + filt_split[-1].strip('()"')
            #     left = filt_split[0]
            #     joinFeature.append([left,right])
            #     hasFilter = False
            # else:
            #     col = alias2table[alias] + '.' + filt_split[0].strip('()"')
                
    except:
        print(filt)
        # import pdb
        # pdb.set_trace()
        # hasFilter = False
        raise ValueError('Column Type Error')
    # if hasFilter:
    #     try:
    #         filterFeature['colName'].append(col)
    #     except:
    #         print(metaInfo['column_type'],filt)
    #         raise ValueError('Column Type Error')
    #     try:
    #         filterFeature["op"].append(OP2IDX[op])
    #     except:
    #         print(f'op = {op} not in OP2IDX', filt)
    #         raise ValueError('op not in OP2IDX')
    #     filterFeature["dtype"].append(dtype)
    #     isInMCV = 0.0
    #     isInHist = 0.0
    #     # if len(val) > 1:
    #     #     print(val)
    #     if len(val) > 0:
    #         for v in val:
    #             if col in metaInfo and 'mcv_vals' in metaInfo[col] and v in metaInfo[col]['mcv_vals']:
    #                 isInMCV += 1
    #             if col in metaInfo and 'hist_vals' in metaInfo[col] and v in metaInfo[col]['hist_vals']:
    #                 isInHist += 1
    #         isInMCV = isInMCV / len(val)
    #         isInHist = isInHist / len(val)
    #     filterFeature['isInMCV'].append(isInMCV)
    #     filterFeature['isInHist'].append(isInHist)
    # print(filt, filterFeature['colName'], joinFeature)

def parse_conditions(planNode, alias, alias2table, metaInfo, filterFeature, joinFeature):
    for condType in CONDTYPE:
        if condType in planNode:
            conditions = planNode[condType]
            conditions = conditions[1:-1]
            and_parts = conditions.split(' AND ')
            i = 0
            while i < len(and_parts):
                part = and_parts[i]
                open_count = 0
                close_count = 0
                in_quotes = False
                for c_s in part:
                    if c_s == "'":
                        in_quotes = not in_quotes
                    elif not in_quotes:
                        if c_s == '(':
                            open_count += 1
                        elif c_s == ')':
                            close_count += 1
                while open_count != close_count and i < len(and_parts) - 1:
                    i += 1
                    part += ' AND ' + and_parts[i]
                    for c_s in and_parts[i]:
                        if c_s == "'":
                            in_quotes = not in_quotes
                        elif not in_quotes:
                            if c_s == '(':
                                open_count += 1
                            elif c_s == ')':
                                close_count += 1

                if ' OR ' in part:
                    or_parts = part.split(' OR ')
                    for or_part in or_parts:
                        if ' AND ' in or_part:
                            and_subparts = or_part.split(' AND ')
                            for subpart in and_subparts:
                                process_condition(subpart.strip('() '), alias,alias2table, metaInfo, filterFeature,joinFeature)

                        else:
                            process_condition(or_part.strip('() '), alias, alias2table, metaInfo, filterFeature,joinFeature)
                else:
                    process_condition(part.strip('() '), alias, alias2table, metaInfo, filterFeature, joinFeature)

                i += 1

class TreeNode:

    def __init__(self ,tableEmbed, typeId, joinEmbed, filters, db_est, pos, gtValue, alias):
        # self.tableSize = tableSize
        self.tableEmbed = tableEmbed
        self.typeId = typeId
        self.joinEmbed = joinEmbed
        self.filterDict = filters
        self.db_est = db_est
        self.pos = pos
        self.alias = alias
        self.gtValue = gtValue
        self.node2feature()

    def node2feature(self):
        # Construct baseline features: typeId, pos (shifted by 1) and 4 database estimates
        baseline = np.concatenate((np.array([self.typeId]), np.array([self.pos + 1]), np.array(self.db_est)))
        # static = np.array([self.isInMcv, self.isInHist])
        if self.filterDict is not None:
            num_filters = len(self.filterDict['op'])
            if num_filters > MAXFILTER:
                print(f'num_filters = {num_filters} > MAXFILTER = {MAXFILTER}, planNode = {self.alias}', flush=True)
                num_filters = MAXFILTER
                self.filterDict['op'] = self.filterDict['op'][:MAXFILTER]
                self.filterDict['dtype'] = self.filterDict['dtype'][:MAXFILTER]
                self.filterDict['isInMCV'] = self.filterDict['isInMCV'][:MAXFILTER]
                self.filterDict['isInHist'] = self.filterDict['isInHist'][:MAXFILTER]
            self.filterDict['op'] = np.pad(np.array(self.filterDict['op']) + 1, (0, MAXFILTER - num_filters), constant_values=0)
            self.filterDict['dtype'] = np.pad(np.array(self.filterDict['dtype']) + 1, (0, MAXFILTER - num_filters), constant_values=0)
            self.filterDict['isInMCV'] = np.pad(np.array(self.filterDict['isInMCV']), (0, MAXFILTER - num_filters), constant_values=0)
            self.filterDict['isInHist'] = np.pad(np.array(self.filterDict['isInHist']), (0, MAXFILTER - num_filters), constant_values=0)
            # filts = np.array(list(self.filterDict.values())) + 1  #cols, ops, dtype
            filts = np.concatenate((self.filterDict['op'], self.filterDict['dtype'], self.filterDict['column'], self.filterDict['isInMCV'], self.filterDict['isInHist']), axis=0)
            filtmask = np.zeros(MAXFILTER)
            filtmask[:num_filters] = 1
        tableEmbed = np.array([self.tableEmbed])
        join = np.array(self.joinEmbed['joinColumn'])
        joinmask = np.array(self.joinEmbed['joinMask'])
        
        # Construct filter features if available. Expecting self.filterDict to contain at least 'colDtype' (for dtype) and optionally 'ops'
        if self.filterDict is not None: 
            self.feature = np.concatenate((baseline, filts, filtmask, tableEmbed, join, joinmask))
        else:
            self.feature = np.concatenate((baseline, tableEmbed, join, joinmask))
        return self.feature
