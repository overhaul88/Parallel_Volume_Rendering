import socket

def is_reachable(host, port=80, timeout=1):
    try:
        socket.create_connection((host, port), timeout)
        return True
    except (socket.timeout, socket.error):
        return False

def generate_reachable_hosts(start_ip, end_ip, port=80, timeout=1):
    reachable_hosts = []

    # Convert IP ranges to integers for easier iteration
    start_ip_int = int(socket.inet_aton(start_ip).hex(), 16)
    end_ip_int = int(socket.inet_aton(end_ip).hex(), 16)

    for ip_int in range(start_ip_int, end_ip_int + 1):
        current_ip = socket.inet_ntoa(bytes.fromhex(hex(ip_int)[2:]))
        if is_reachable(current_ip, port, timeout):
            reachable_hosts.append(current_ip)

    return reachable_hosts

# Example usage: Generate a list of reachable hosts in a specific IP range
start_ip = "172.27.19.1"
end_ip = "172.27.19.60"
reachable_hosts = generate_reachable_hosts(start_ip, end_ip)

print("Reachable hosts:")
for host in reachable_hosts:
    print(host)
