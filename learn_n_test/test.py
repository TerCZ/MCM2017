class A:
    def __init__(self, data):
        self.data = data


a = A(1)
b = A(2)
c = A(0)

l = [a, b, c]
l.sort(key=lambda x:x.data)
print([x.data for x in l])