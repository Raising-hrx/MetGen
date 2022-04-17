import copy

def pre_process(sent):
    if sent.endswith('.'):
        return sent
    else:
        return sent+'.'

def un_pre_process(sent):
    if sent.endswith('.'):
        return sent[:-1]
    else:
        return sent


    
def chunk(it, n):
    c = []
    for x in it:
        c.append(x)
        if len(c) == n:
            yield c
            c = []
    if len(c) > 0:
        yield c

# -----utils for parse annotated data and predicted trees -----
def get_sent_list(item): return item['meta']['triples'] # return dict {sid:sent}
def get_int_list(item): return item['meta']['intermediate_conclusions']
def get_hid(item): return item['meta']['hypothesis_id']

def parse_proof(proof):
    step_proof = []
    for step in proof.split('; '):
        if not step: continue
        step = step.split(':')[0]
        tmp = [step.split(' -> ')[0].replace(' ','').split('&'),
              step.split(' -> ')[1].replace(' ','')]
        step_proof.append(tmp)
            
    return step_proof

def get_node(idx,node_list):
    """
    find the node with id from the node_list
    """
    node = None
    for item in node_list:
        if item['id'] == idx:
            node = item
    return node

def get_tree(idx,node_list):
    """
    rerank the node_list
    make the idx node as root node
    only retrain nodes in the tree
    """
    childrens = []
    node = get_node(idx,node_list)
    
    if node:
        for child_idx in node['pre']:
            childrens += get_tree(child_idx,node_list)

        return [node] + childrens
    else:
        return None
    
def get_gt_node_list(item):
    """
    load the ground truth tree from the orignal dataset item
    add full stop
    """
    node_list = []
    for sent_id, sent in get_sent_list(item).items():
        if sent_id.startswith('sent'):
            node_list.append({
                'id':sent_id,
                'sent':pre_process(sent),
                'pre':[],
            })

    step_proof = parse_proof(item['meta']['step_proof'])
    for sent_id, sent in get_int_list(item).items(): 
        if sent_id == item['meta']['hypothesis_id']:
            index = [step[1] for step in step_proof].index('hypothesis')
        else:
            index = [step[1] for step in step_proof].index(sent_id)
        node_list.append({
            'id':sent_id,
            'sent':pre_process(sent),
            'pre':step_proof[index][0],
        })   
        
    node_list = get_tree(item['meta']['hypothesis_id'],node_list)
        
    return node_list

def get_leaves_ids(idx,node_list):
    """
    get all leaf nodes in the tree (sentX / pre in none)
    """
    leaves_ids = []
    node = get_node(idx,node_list)
    
    if node:
        if node['id'].startswith('sent') or len(node['pre']) == 0:
            return [node['id']]
        else:
            for child_idx in node['pre']:
                leaves_ids += get_leaves_ids(child_idx,node_list)
                
        return leaves_ids
    else:
        return []

def get_all_leaves_ids(node_list):
    return [node['id'] for node in node_list if node['id'].startswith('sent')]

def print_node_tree(node_idx,node_list,depth = 0):
    node = get_node(node_idx,node_list)
    if depth == 0:
        print(node['id']+': '+node['sent'])
    else:
        print('\t| '*(depth-1)+'\t|- '+node['id'] +': '+node['sent'])
    for child_idx in node['pre']:
        print_node_tree(child_idx,node_list,depth+1)
        
def print_node_tree_with_type(node_idx,node_list,depth = 0):
    node = get_node(node_idx,node_list)
    if depth == 0:
        print(f"{node['id']}: {node['sent']} ({node.get('step_type','None')}  pred: {node.get('orig_sent','')}) ")
    else:
        print('\t| '*(depth-1)+'\t|- '+ f"{node['id']}: {node['sent']} ({node.get('step_type','None')}) ")
    for child_idx in node['pre']:
        print_node_tree_with_type(child_idx,node_list,depth+1)
        

def print_node_tree_str(node_idx,node_list,depth = 0):
    node = get_node(node_idx,node_list)
    if depth == 0:
        s = f"{node['id']}: {node['sent']}"
    else:
        s = '\t| '*(depth-1)+'\t|- '+node['id'] +': '+node['sent']
    for child_idx in node['pre']:
        s_child = print_node_tree_str(child_idx,node_list,depth+1)
        s = s + '\n' + s_child
        
    return s
    
def convert_to_result(tree, data_item, make_new_int = False):
    # tree -> csv result
    # tree: predict node list
    # data_item: gold node list
    if tree is None:
        result = {
            'id' : data_item['id'],
            'steps' : [],
            'texts' : {},
            'proof' : '',
        }
        return result, {}
    
    tree = copy.deepcopy(tree)
    
    if make_new_int:
        # make new intermediate sentences id
        new_int_id_dict = {}
        new_counter = 0
        for node in tree:
            if node['id'].startswith('sent'):
                new_int_id_dict[node['id']] = node['id']
            else:
                new_int_id_dict[node['id']] = f"int{new_counter}"
                new_counter += 1 

        for node in tree:
            node['id'] = new_int_id_dict[node['id']]
            node['pre'] = [new_int_id_dict[idx] for idx in node['pre']]
        
    # convert tree to result item
    result = {}
    result['id'] = data_item['id']
    result['steps'] = []
    result['texts'] = {}
    
    root = tree[0]
    result['texts']['hypothesis'] = data_item['hypothesis']
    result['steps'].append([root['pre'],'hypothesis'])
    
    for node in tree[1:]:
        if node['id'].startswith('sent'):
            result['texts'][node['id']] = node['sent']
        else:
            result['texts'][node['id']] = node['sent']
            result['steps'].append([node['pre'],node['id']])
        
    proof = ''
    for step_pre, step_con in result['steps'][::-1]:
        proof += " & ".join(step_pre) + " -> " + step_con
        if step_con != 'hypothesis':
            proof += ': '+ un_pre_process(result['texts'][step_con]) +'; '
        else:
            proof += '; '
            
    result['proof'] = proof

    return result

def rename_node(tree, id_map = None):
    # rename the node id of the tree given id_map
    # if id_map is None, id_map{'x'} = 'rename_x'
    if id_map is None:
        id_map = {}
        for node in tree:
            nid = node['id']
            new_nid = 'rename_' + nid
            id_map[nid] = new_nid
    else:
        for node in tree:
            nid = node['id']
            if nid not in id_map.keys():
                id_map[nid] = nid
    
    new_tree = copy.deepcopy(tree)
    for node in new_tree:
        if node['id'] in id_map:
            node['id'] = id_map[node['id']]
        node['pre'] = [id_map[nid] if nid in id_map else nid for nid in node['pre']]
    return new_tree


# -----Evaluation utils -----
def Jaccard(set1,set2):
    set1 = set(set1)
    set2 = set(set2)
    Intersection = len(set1.intersection(set2))
    Union = len(set1.union(set2))
    
    return Intersection / (Union + 1e-20)

def div(num, denom):
    return num / denom if denom > 0 else 0

def compute_f1(matched, predicted, gold):
    # 0/0=1; x/0=0
    precision = div(matched, predicted)
    recall = div(matched, gold)
    f1 = div(2 * precision * recall, precision + recall)
    
    if predicted == gold == 0:
        precision = recall = f1 = 1.0
    
    return precision, recall, f1



