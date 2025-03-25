import dpkt

max_byte_len = 12

categories = ["Chat", "Email"]
labels = [0, 1]


def gen_pkts(pcap, label):
    pkts = {}

    if pcap.datalink() != dpkt.pcap.DLT_EN10MB:
        print('unknown data link!')
        return

    pkts[str(label)] = []

    for _, buff in pcap:

        eth = dpkt.ethernet.Ethernet(buff)

        if isinstance(eth.data, dpkt.ip.IP) and (
                isinstance(eth.data.data, dpkt.udp.UDP)
                or isinstance(eth.data.data, dpkt.tcp.TCP)):
            ip = eth.data
            pkts[str(label)].append(ip)

    return pkts


def closure(pkts):

    pkts_dict = {}
    for name in categories:
        index = categories.index(name)
        pkts_dict[name] = pkts[index]
    
    return pkts_dict

def pkt2feature(data):
    data_list = []

    for c in categories:
        all_pkts = []
        p_keys = list(data[c].keys())

        for key in p_keys:
            all_pkts = data[c][key]
            print(len(all_pkts))


        for idx in range(len(all_pkts)):
            pkt = all_pkts[idx]
            raw_byte = pkt.pack()

            byte = []
            pos = []
            for x in range(min(len(raw_byte), max_byte_len)):
                byte.append(int(raw_byte[x]))
                pos.append(x)

            byte.extend([0] * (max_byte_len - len(byte)))
            pos.extend([0] * (max_byte_len - len(pos)))
            byte.append(c)
            
            data_list.append(byte)
            
            
    return data_list


pkts_list = []

chat = dpkt.pcap.Reader(open('testchat.pcap', 'rb'))
chat_dic = gen_pkts(chat, 0)
pkts_list.append(chat_dic)

email = dpkt.pcap.Reader(open('testemail.pcap', 'rb'))
email_dic = gen_pkts(email, 1)
pkts_list.append(email_dic)

data = closure(pkts_list)

IPbytes = pkt2feature(data)

print(IPbytes)