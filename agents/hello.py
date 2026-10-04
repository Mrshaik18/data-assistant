nums=[5,7,8,4,1,6,9,2]

def sort_num(nums):
    for i in range(0,len(nums)):
        mini_idx=i
        for j in range(i+1,len(nums)):
            if nums[j]<nums[mini_idx]:
                mini_idx=j
        nums[i],nums[mini_idx]=nums[mini_idx],nums[i]

sort_num(nums)

print(nums)