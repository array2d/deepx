import socket
from typing import Optional, Tuple
import select

class UDPConn:
    def __init__(self, endpoint: str = "localhost:9090"):
        # 解析endpoint
        self._host, port_str = endpoint.split(':')
        self._port = int(port_str)
        # 创建UDP socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 设置非阻塞模式
        self._sock.setblocking(False)
        # 设置接收缓冲区
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
    
    def send(self, ir: str) -> Optional[dict]:
        
        # 发送IR字符串
        try:
            # 将IR字符串编码为bytes并发送
            data = ir.encode('utf-8')
            self._sock.sendto(data, (self._host, self._port))
            # 等待响应
            return self._wait_response()
            
        except Exception as e:
            print(f"发送IR失败: {e}")
            return None

    def _wait_response(self, timeout: float =10000) -> any:
        """等待并接收响应
        
        Args:
            timeout: 超时时间(秒)
        """
        try:
            # 使用select实现超时等待
            ready = select.select([self._sock], [], [], timeout)
            if ready[0]:
                data, addr = self._sock.recvfrom(65536)  # 64KB缓冲区
                response = data.decode('utf-8')
                return response
            return None
            
        except Exception as e:
            print(f"接收响应失败: {e}")
            return None

    def __del__(self):
        """确保socket正确关闭"""
        if hasattr(self, '_sock'):
            self._sock.close()

# 全局单例实例
_default_udpconn = UDPConn()
