# 升序有序数组，用二分查找元素定位
def binarysearch(data:tuple[int],target:int):
    left,right=0,len(data)-1
    while left<=right:
        mid=(left+right)//2
        if data[mid]<target:
            left=mid+1
        elif data[mid]==target:
            return mid
        else:
            right=mid-1
    return -1

test=(1,2,5,6,7,8,9,12,14)
print(binarysearch(test,8))