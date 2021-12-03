"""
Data Mining
Project, Development and Evaluation of Graph Clustering Algorithms
Myeong suhwan
"""

from datetime import datetime
import sys
import matplotlib.pyplot as plt


def getDataFromFile(filename):
    input_file = open(filename, "r")
    graph = dict()

    line = None
    line = input_file.readline().split()

    while line != "":

        if not line[0] in graph:
            graph[line[0]] = [line[1]]

        elif line[0] in graph:
            connected = graph[line[0]]
            connected.append(line[1])
            graph[line[0]] = list(set(connected))

        if not line[1] in graph:
            graph[line[1]] = [line[0]]

        elif line[1] in graph:
            connected = graph[line[1]]
            connected.append(line[0])
            graph[line[1]] = list(set(connected))

        line = input_file.readline().split()
        if not line:
            break
    # print(graph)

    return graph


def output_to_file(filename, cluster):
    file = open(filename, "w")

    for v in cluster:
        if not v:
            continue
        
        string = " ".join(v)
        file.write("{0}: {1}".format(len(v), string))
        file.write("\n")
    file.close()


def GetNeighbors(graph, vertex, node_set=set()):
    
    for r in graph[vertex]:
        node_set.add(r)
    #print("node set : ",node_set)
    return node_set


def GetJaccardCoefficient(graph, node1, node2):

    S = len(
        GetNeighbors(graph, node1, node_set=set()).intersection(
            GetNeighbors(graph, node2, node_set=set())
        )
    ) / len(
        GetNeighbors(graph, node1, node_set=set()).union(
            GetNeighbors(graph, node2, node_set=set())
        )
    )

    return S


def GetDensity(graph, edge_list):
    vertexNum = len(graph)
    possibleEdgeNum = int(vertexNum * (vertexNum - 1) / 2)
    edges_len = len(edge_list)
    # print("edges len :", edges_len)
    # print("possible all edge len : ", possibleEdgeNum)
    density = edges_len / possibleEdgeNum
    # print("density : ", density)

    return density


def GetEdgeList(graph):
    edge_list = list()

    for v in graph:
        # print(v ," ::", len(graph[v]),graph[v])
        for node2 in graph[v]:
            tmp = {v, node2}
            edge_list.append(tmp)
    # print("edge list ", edge_list)

    new_edge_list = list()

    for edge in edge_list:
        if edge not in new_edge_list:
            new_edge_list.append(edge)

    return new_edge_list


def GetSubGraph(graph, cluster):
    new_graph = dict()

    for vertex in cluster:
        if vertex in graph:
            new_graph[vertex] = graph[vertex]

    return new_graph


def MakeIncidentMatrix(file):
    input_file = open(file, 'r')
    incident_list = list()
    

    for line in input_file:
        data = line.split()
        incident_list.append(data[1:])
        
    
    C_dict = dict()
    for cluster in incident_list:
        for node1 in cluster:
            for node2 in cluster:
                if node1 == node2:
                    continue
                C_dict[node1,node2] = 1 #* same cluster
    #! same cluster가 아닌건 아예 딕셔너리에 없음.
    
    
    return C_dict

    

def GetGroundTruth(input_file):
    file = open(input_file, 'r')
    ground_truth_list = list()


    
    for line in file:
        data = line.split()
        ground_truth_list.append(data) # cluster 단위
        
    P_dict = dict()
    for cluster in ground_truth_list:
        for node1 in cluster:
            for node2 in cluster:
                if node1 == node2:
                    continue
                P_dict[node1,node2] = 1 #* same cluster
    #! same cluster가 아닌건 아예 딕셔너리에 없음.
    
    
    return P_dict, ground_truth_list

def EvaluationJaccard(C_dict,P_dict):
    SS = 0
    SD = 0
    DS = 0
    # print("P dict : ", P_dict)
    # print("C dict : ", C_dict)
    for node_pair in P_dict:
        #print("node pair : ",node_pair)
        if node_pair in C_dict:
            if P_dict[node_pair] == 1 and C_dict[node_pair] == 1:
                SS += 1
        if not node_pair in C_dict:
            DS += 1

    for node_pair in C_dict:
        if not node_pair in P_dict:
            SD += 1
    
            

    # print("SS : ",SS)
    # print("SD : ",SD)
    # print("DS : ",DS)
    
    #print("[+]Calculating the accuracy using Jaccard Index . . .")
    accuracy = (SS / (SS+SD+DS)) * 100
    
    return accuracy

def newMethod(graph):

    #* 그래프에서 두 개의 노드를 선정
    #* neighbor set을 구함
    #* group based method - jaccard index로 similarity를 구함.
    #* similarity가 0.5 이상이면 same cluster로 분류
    tmp_graph = graph.copy()
    node_sim = list()
    redundant_list = list()
    cnt = 0
    for node1 in graph:
        if node1 not in tmp_graph:
            continue
        cnt += 1
        print(node1 ,"cnt : ",cnt," of ",len(graph))
        #for node2 in graph:
        for node2 in graph[node1]:
            if node1 == node2:
                continue
            if {node1,node2} in redundant_list:
                continue
            N1 = GetNeighbors(graph,node1,node_set=set(node1))
            #print(node1 ," 의 neighbor : ", N1)
            N2 = GetNeighbors(graph,node2,node_set=set(node2))
            #print(node2 ," 의 neighbor : ", N2)
            sim = GetSimilarityUsingGroupBasedMethod(N1,N2)
            tmp_list = [node1,node2,sim]
            # if [node2,node1,sim] in node_sim:
            #     continue
            node_sim.append(tmp_list)
            redundant_list.append({node1,node2})
            #print(tmp_list)
        #del tmp_graph[node1]
    #node_sim = list(set([tuple(set(item)) for item in node_sim])) -> 데이터순서꼬임
    for v in node_sim:
        print(v)
    print("len of nodesim : ",len(node_sim))
    # new_node_similarity = list()
    # count = 0
    # for v in node_sim:
    #     count += 1
    #     print("count : ", count, " of ",len(node_sim))
    #     if [v[1],v[0],v[2]] in new_node_similarity or [v[0],v[1],v[2]] in new_node_similarity:
    #         continue
    #     else:
    #         #print(v,"를 추가하였습니다!")
    #         new_node_similarity.append(v)

    #print("len of new nodesim : ",len(new_node_similarity))
    return node_sim

def MakeCluster(node_sim):

    cluster = list()
    for element in node_sim:
        #print("element : ", element)
        node1 = element[0]
        node2 = element[1]
        similarity = element[2]
        if similarity >= 0.2: #! 조절할부분
            cluster.append({node1,node2})
            
    
    isChanged = True
    count = 1
    while isChanged:
        print("count : ", count)
        count+=1
        isChanged = False
        
        for clst in cluster:
            
            for next_clst in cluster:
                if clst == next_clst:
                    continue
                
                if clst.intersection(next_clst):
                    cluster[cluster.index(clst)] = clst.union(next_clst)
                    clst = clst.union(next_clst)
                    cluster[cluster.index(next_clst)] = set()
                    next_clst = set()
                    isChanged = True
    
    new_cluster = list()
    for v in cluster:
        if v and v not in new_cluster:
            new_cluster.append(v)
    
    return new_cluster




def GetSimilarityUsingGroupBasedMethod(gene1_ancestor, gene2_ancestor):
    union_set = gene1_ancestor.union(gene2_ancestor)
    
    inter_set = gene1_ancestor.intersection(gene2_ancestor)
    # print("union : ",union_set)
    # print("inter : ",inter_set)
    return float(len(inter_set)) / float(len(union_set))

def main():
    # inputfile = 'assignment5_input.txt' #! sys
    # inputfile = 'test_input.txt'

    if len(sys.argv) != 3:
        print("No input file.")
        print("<Usage> project.py project_input.txt complex_merged.txt")
        # * inputfile = 'test_input.txt'
        return -1

    if len(sys.argv) == 3:
        inputfile = sys.argv[1]
        groundtruth_file = sys.argv[2]
    output_filename = "output.txt"
    
    graph = getDataFromFile(inputfile)
    start_time = datetime.now()
    
    
    


    node_sim_list = newMethod(graph)
    similarity = list()
    for v in node_sim_list:
        similarity.append(v[2])
        
    print("new method end")
    cluster = MakeCluster(node_sim_list)

    # * save to output file with size : nodes in a cluster
    output_to_file(output_filename, cluster)
    P_dict, ground_truth_list = GetGroundTruth(groundtruth_file)
    C_dict = MakeIncidentMatrix(output_filename)
    accuracy = EvaluationJaccard(C_dict,P_dict)
    print("accuracy : ",accuracy,"%")

    # * f-measure
    f_score_list = list()
    for X in cluster:
        f_score = 0
        for Y in ground_truth_list:
            if not X.intersection(Y):
                continue
            recall = len(X.intersection(Y)) / len(Y)
            precision = len(X.intersection(Y)) / len(X)
            tmp = 2 * recall * precision / (recall+precision)
            if tmp >= f_score:
                f_score = tmp
        f_score_list.append(f_score)

    f_score_avg = sum(f_score_list) / len(f_score_list)
    print("f-Score Average : ", f_score_avg)
    
    #print(len(similarity))
    plt.hist(similarity)
    plt.show()


    print("[+] Time Elapsed : ", datetime.now() - start_time, "microseconds")

if __name__ == "__main__":
    main()
