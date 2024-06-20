import networkx as nx
import random

# 노선별 역 정의
subway_lines = {
    '2호선': ['강남', '역삼', '선릉', '삼성', '종합운동장', '잠실새내', '잠실', '잠실나루', '강변', '구의', '건대입구', '성수', '뚝섬', '한양대', '왕십리', '상왕십리', '신당', '동대문역사문화공원', '을지로4가', '을지로3가', '을지로입구', '시청', '충정로', '아현', '이대', '신촌', '홍대입구', '합정', '당산', '영등포구청', '문래', '신도림', '대림', '구로디지털단지', '신대방', '신림', '봉천', '서울대입구', '낙성대', '사당', '방배', '서초', '교대', '강남'],
    '3호선': ['수서', '일원', '대청', '학여울', '대치', '도곡', '매봉', '양재', '남부터미널', '교대', '고속터미널', '잠원', '신사', '압구정', '옥수', '금호', '약수', '동대입구', '충무로', '을지로3가', '종로3가', '안국', '경복궁', '독립문'],
    '6호선': ['마포구청', '망원', '합정', '상수', '광흥창', '대흥', '공덕', '효창공원앞', '삼각지', '녹사평', '이태원', '한강진', '버티고개', '약수', '청구', '신당', '동묘앞', '창신', '보문', '안암', '고려대'],
    '수인분당선': ['수서', '대모산입구', '개포동', '구룡', '도곡', '한티', '선릉', '선정릉', '강남구청', '압구정로데오', '서울숲', '왕십리', '청량리'],
    '9호선': ['개화', '김포공항', '공항시장', '신방화', '마곡나루', '양천향교', '가양', '증미', '등촌', '염창', '신목동', '선유도', '당산', '국회의사당', '여의도', '샛강', '노량진', '노들', '흑석', '동작', '구반포', '신반포', '고속터미널', '사평', '신논현', '언주', '선정릉', '삼성중앙', '봉은사', '종합운동장', '삼전', '석촌고분', '석촌', '송파나루', '한성백제', '올림픽공원','둔촌오륜', '중앙보훈병원'],
    

}

# 환승역 및 환승 시간 정의 (2~6분)
transfer_stations = {
    '선릉': [('2호선', '수인분당선', random.randint(2, 6))],
    '왕십리': [('2호선', '수인분당선', random.randint(2, 6))],
    '을지로3가': [('2호선', '3호선', random.randint(2, 6))],
    '도곡': [('3호선', '수인분당선', random.randint(2, 6))],
    '합정': [('2호선', '6호선', random.randint(2, 6))],
    '신당': [('2호선', '6호선', random.randint(2, 6))],
    '약수': [('3호선', '6호선', random.randint(2, 6))],
    '수서': [('3호선', '수인분당선', random.randint(2, 6))],
    '당산': [('9호선', '2호선', random.randint(2, 6))],
    '고속터미널': [('3호선', '9호선', random.randint(2, 6))],
    '선정릉' : [('9호선', '수인분당선', random.randint(2, 6))],
    '종합운동장' : [('9호선', '2호선', random.randint(2, 6))],
    '교대' : [('3호선', '2호선', random.randint(2, 6))]
}

# 그래프 생성
G = nx.Graph()

# 노드 추가
for line, stations in subway_lines.items():
    for station in stations:
        G.add_node(f"{station}_{line}", station=station, line=line)

# 엣지 추가 (기본 3분 거리)
for line, stations in subway_lines.items():
    for i in range(len(stations) - 1):
        G.add_edge(f"{stations[i]}_{line}", f"{stations[i + 1]}_{line}", weight=3)

# 환승 엣지 추가 (랜덤 환승 시간 적용)
for station, transfers in transfer_stations.items():
    for from_line, to_line, transfer_time in transfers:
        G.add_edge(f"{station}_{from_line}", f"{station}_{to_line}", weight=transfer_time)

def find_route(start, end):
    if start == end:
        return "같은 역입니다.", []

    # 시작역과 도착역이 어느 노선에 속하는지 확인
    start_nodes = [n for n in G.nodes if G.nodes[n]['station'] == start]
    end_nodes = [n for n in G.nodes if G.nodes[n]['station'] == end]

    shortest_path = None
    min_time = float('inf')

    for start_node in start_nodes:
        for end_node in end_nodes:
            try:
                path = nx.dijkstra_path(G, start_node, end_node, weight='weight')
                time = nx.dijkstra_path_length(G, start_node, end_node, weight='weight')
                if time < min_time:
                    min_time = time
                    shortest_path = path
            except nx.NetworkXNoPath:
                continue

    if shortest_path is None:
        return "해당 역들 사이의 경로를 찾을 수 없습니다.", []

    # 환승 정보 추출
    transfers = []
    for i in range(1, len(shortest_path) - 1):
        current_station = G.nodes[shortest_path[i]]['station']
        current_line = G.nodes[shortest_path[i]]['line']
        previous_line = G.nodes[shortest_path[i - 1]]['line']
        next_line = G.nodes[shortest_path[i + 1]]['line']
        if previous_line != current_line:
            transfers.append((current_station, previous_line, current_line))
        if current_line != next_line:
            transfers.append((current_station, current_line, next_line))

    # 중복된 환승 정보를 제거하고 형식에 맞게 정리
    unique_transfers = []
    seen = set()
    for transfer in transfers:
        if (transfer[0], transfer[1], transfer[2]) not in seen:
            seen.add((transfer[0], transfer[1], transfer[2]))
            unique_transfers.append(transfer)

    if unique_transfers:
        transfer_info = ', '.join([f"{t[0]}에서 {t[2]}으로" for t in unique_transfers])
        return f"총 {min_time}분이 걸리고, 환승은 {transfer_info} 해야 합니다.", shortest_path
    else:
        return f"총 {min_time}분이 걸립니다.", shortest_path

def main():
    start = input("출발역: ")
    end = input("도착역: ")

    valid_stations = set()
    for stations in subway_lines.values():
        valid_stations.update(stations)

    if start not in valid_stations or end not in valid_stations:
        print("잘못된 역입니다.")
    else:
        message, path = find_route(start, end)
        print(message)

if __name__ == "__main__":
    main()
