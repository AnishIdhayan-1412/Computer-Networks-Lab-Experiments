# Computer-Networks-Lab-Experiments
Computer Networks Lab Experiments


This repo has my work for the Computer Networks lab experiments using Cisco Packet Tracer.  
I’ve added the `.pkt` files and screenshots of the results.

---

## Lab 1 – Peer to Peer Connection + Study of Cables

**Aim:**  
Connect two PCs directly and check if they can communicate. Also note down the cable color codes.

**Steps I followed:**
- Dropped two PCs in Packet Tracer and connected them with a copper cross-over cable.
- Gave IPs:
  - PC0 → `192.168.1.1` / `255.255.255.0`
  - PC1 → `192.168.1.2` / `255.255.255.0`
- Pings worked fine between them.
- Then I added **two more PCs** and a **switch**.
- Connected those two extra PCs to the switch using straight-through cables.
- Gave IPs in the same subnet:
  - PC2 → `192.168.1.3` / `255.255.255.0`
  - PC3 → `192.168.1.4` / `255.255.255.0`
- Checked connectivity — all four PCs could ping each other successfully.

**Cable types I studied:**
- **Straight-Through** – for different devices (PC to Switch).
- **Cross-Over** – for same type devices (PC to PC).
- **Rollover** – for console connections.

**T568A & T568B color codes:**

| Pin | T568A        | T568B        |
|-----|--------------|--------------|
| 1   | White/Green  | White/Orange |
| 2   | Green        | Orange       |
| 3   | White/Orange | White/Green  |
| 4   | Blue         | Blue         |
| 5   | White/Blue   | White/Blue   |
| 6   | Orange       | Green        |
| 7   | White/Brown  | White/Brown  |
| 8   | Brown        | Brown        |

**Result:**  
Ping was successful between all devices.

---

## Lab 2 – Network Topologies

**Aim:**  
Create and test different topologies in Packet Tracer.

**What I did:**

**1. Star Topology**
- One switch in the middle, three PCs around it.
- Straight-through cables from each PC to switch.
- IPs in same subnet, pings worked.

**2. Bus Topology (using a Hub)**
- Hub in middle, three PCs connected with straight-through.
- Same subnet IPs, pings worked fine.

**Result:**  
Both topologies worked without any issues.

---
