# tcpraplay

```shell
cd /home/solana/twctest_data/pcaps
sudo tcpreplay -i enp1s0f0np0 -l 0 -t --duration 300 --unique-ip netflix_50mb.pcap

sudo tcpreplay -i enp1s0f0np0 -l 0 --mbps 2048 --duration 300 --unique-ip netflix_50mb.pcap

sudo tcpreplay -i enp1s0f0np0 -l 0 -t --duration 300 --unique-ip --unique-ip-loops 5 netflix_50mb.pcap
nload -t 100 -u g -U G devices enp1s0f0np0

ip link set enp3s0f0 promisc on

```




## Model Commands

```shell
# File based extraction
twc extract --mode file -m 40app_models_1-6-0-24_v4_live.json --sequence-length 150 --payload-bytes 256 --flow-duration 60 --min-flow-packets 1 --max-flow-packets 5000 --recent-flow-duration 10 --notimestamp --no-name-cache <pcap_file>

# Live extraction
twc extract --mode live -m 40app_models_1-6-0-24_v4_live.json --sequence-length 150 --payload-bytes 256 --flow-duration 60 --min-flow-packets 1 --max-flow-packets 5000 --recent-flow-duration 10 --notimestamp --no-name-cache <interface_name>
```

# Experiments

```text
task 1: 5 min/10g with unique ip     task1_5_10g_unique_ip                  ok
task 2: 5 min/10g without unique ip  task2_5_10g_no_unique_ip               ok
task 3: 1 min/10g with unique ip    task3_1_10g_unique_ip                   ok  
task 4: 1 min/10g without uniue ip   task4_1_10g_no_unique_ip               ok

task 5: 1 min/2g with unique ip    task5_1_2g_unique_ip                     ok
task 6: 1 min/4g with unique ip    task6_1_4g_unique_ip
task 7: 1 min/6g with unique ip    task7_1_6g_unique_ip
task 8: 1 min/8g with unique ip    task8_1_8g_unique_ip

task 9: 1 min/2g without unique ip    task9_1_2g_no_unique_ip
task 10: 1 min/4g without unique ip    task10_1_4g_no_unique_ip
task 11: 1 min/6g without unique ip   task11_1_6g_no_unique_ip
task 12: 1 min/8g without unique ip   task12_1_8g_no_unique_ip

```