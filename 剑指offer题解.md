## 剑指offer题解

#### 03-数组中重复的数字

最容易想到的利用set不重复的特性，这里还可以注意HashSet的用法。

```java
class Solution {
    public int findRepeatNumber(int[] nums) {
        Set<Integer> dic = new HashSet<>();
        for(int num:nums){
            if(dic.contains(num)) return num;
            else dic.add(num);
        }
        return -1;
    }
}
```

第一种做法需要额外空间，还有一种更巧妙的原地交换的做法。可以这么做的原因是所有数字都在0~n-1的范围内，因此可以根据坐标来判断。

```java
class Solution {
    public int findRepeatNumber(int[] nums) {
        int i = 0;
        while(i < nums.length) {
            if(nums[i] == i) {
                i++;
                continue;
            }
            if(nums[nums[i]] == nums[i]) return nums[i];
            int tmp = nums[i];
            nums[i] = nums[tmp];
            nums[tmp] = tmp;
        }
        return -1;
    }
}
```

#### 04-二维数组查找

二维数组往左和往下都是递增，那么从右上角开始遍历，大了就往左，小了就往右，有点类似一个root为右上元素的二叉排序树，很巧妙。

```Java
class Solution {
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        if(matrix.length==0) return false;
        int rows = matrix.length, columns = matrix[0].length;
        int i = 0;
        int j = columns - 1;
        while(j >= 0 && i <= rows-1){
            if(matrix[i][j] == target) return true;
            else if(matrix[i][j] < target) i++;
            else if(matrix[i][j] > target) j--;
        }
        return false;
    }
}
```

#### 05-替换空格

java的字符串不可变，所以用java的话主要是学习一下StringBuilder和StringBuffer的用法，当然工作中直接用replace完事，C++可以遍历一个个去修改长度，但感觉意义不大。

#### 06-从尾到头打印链表

利用递归，这里有一个技巧是在递归最后一层的时候初始化数组，用i，j两个变量来控制。

本题也可显式调用栈，原理相同。

```Java
class Solution {
    int [] res;
    int i=0;
    int j=0;
    public int[] reversePrint(ListNode head) {
        recurv(head);
        return res;
    }
    void recurv(ListNode head){
        if(head==null) {
            res = new int[i];
            return;
        }
        i++;
        recurv(head.next);
        res[j] = head.val;
        j++;
    }
}
```

#### 07-重建二叉树

同leetcode105，这里新学到可用HashMap存储中序序列，可以快速找到根节点在中序中的位置，不需要每次都遍历。

```Java
class Solution {
    private Map<Integer,Integer> indexMap;

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int n = preorder.length;
        indexMap = new HashMap<Integer, Integer>();
        for (int i = 0; i < n; i++) {
            indexMap.put(inorder[i], i);
        }
        return build(preorder, inorder, 0, n - 1, 0, n - 1);

    }
    public TreeNode build(int[] preorder,int[] inorder, int pre_left, int pre_right, int in_left,int in_right){
        if(pre_left>pre_right) return null;
        int pre_root = pre_left;
        int in_root = indexMap.get(preorder[pre_root]);
        TreeNode root = new TreeNode(preorder[pre_root]);
        //左子树节点数目
        int left_tree_size = in_root-in_left;
        root.left = build(preorder, inorder,pre_left+1,pre_left+left_tree_size,in_left,in_root-1);
        root.right = build(preorder, inorder,pre_left+left_tree_size+1,pre_right,in_root+1,in_right);
        return root;
    }

}
```

#### 09-两个栈模拟队列

s1负责入队，s2负责出队，记住原理就好，有一个疑问是题解有些用的是LinkedList当栈用，为什么不用Stack类呢？

```java
class CQueue {
    private Stack<Integer> s1;
    private Stack<Integer> s2;
    public CQueue() {
        s1 = new Stack<Integer>();
        s2 = new Stack<Integer>();
    }
    
    public void appendTail(int value) {
        s1.push(value);
    }
    
    public int deleteHead() {
        if(!s2.empty()) return s2.pop();
        while(!s1.empty()){
            s2.push(s1.pop());
        }
        if(!s2.empty()) return s2.pop();
        else return -1;
    }
}
```

#### 10-1斐波那契

递归比较慢，用最简单的动态规划。

```java
class Solution {
    public int fib(int n) {
        int a=0,b=1,sum=0;
        for(int i=0;i<n;i++){
            sum = (a+b)%1000000007;
            a = b;
            b = sum;
        }
        return a;
    }
}
```

#### 10-2跳台阶

和斐波那契只是初始数字不同。

#### 11 旋转数组的最小数字

第一想法是遍历找到递增变递减的地方，这种做法是O(n)的复杂度，而二分查找可以做到O(log2n)。

leetcode官方题解写的非常清楚，数形结合看根据mid和high比较的三种情况。

```Java
class Solution {
    public int minArray(int[] numbers) {
        int low = 0, high = numbers.length-1;
        while(low < high){
            int mid = low + (high - low) / 2;
            if(numbers[mid] < numbers[high]){
                high = mid;
            }else if(numbers[mid] > numbers[high]){
                low = mid + 1;
            }else if(numbers[mid] == numbers[high]){
                high--;
            }
        }
        return numbers[low];
    }
}
```

二分查找极其重要，再多刷几题感受一下吧。

#### 53-1 在排序数组中查找数字（一）

这题拿到第一反应是二分查找找到其中一个target的位置，再前后遍历去找所有的target，还是改不了喜欢遍历的毛病啊。

正确解法是两次二分查找，分别查找target区间的左边界和右边界。

这里涉及到一个二分查找搜索边界的问题，在写法细节上和二分查找搜索单个元素有所不同。

```Java
class Solution {
    public int search(int[] nums, int target) {
        if(nums.length==0) return 0;
        int low = 0, high = nums.length-1;
        while(low<=high){
            int mid = low + (high - low) / 2;
            if(nums[mid] <= target){
                low = mid + 1;
            }else if(nums[mid] > target){
                high = mid - 1;
            }
        }
        int right = low;
        if(high>=0 && nums[high]!=target) return 0;  //注意这里，右边界不等于target意味着数组中不包含target，直接返回0
        low = 0;
        high = right;
        while(low<=high){
            int mid = low + (high - low) / 2;
            if(nums[mid] < target){
                low = mid + 1;
            }else if(nums[mid] >= target){
                high = mid - 1;
            }
        }
        int left = high;
        return right - left - 1;
    }
}
```

两段二分查找可以合并成同一个函数以减少代码量，但是写成两段逻辑更清晰一点，姑且就先这样吧。

#### 53-2 0~n-1中缺失的数字

到这题对二分查找的思路就比较清晰了，值得注意的有以下几个地方

1、left<right 还是left<=right

这里判断的依据是while循环应该在left=right还是left==right+1。具体来说就是应该在[2,2]就停止还是要到[3,2]，可以带值进去具体思考一下。

2、缩小区间的时候mid是否要在剩余区间内？

判断依据是此时mid还有没有可能是你最终要找的位置，如果有可能，就要包含在内。

3、最终return的是left还是right，如果1中是写left<right，那么结束时l=r，返回哪个都一样，否则还是带值具体分析吧。

此题题解，思路较简单不再赘述

```Java
class Solution {
    public int missingNumber(int[] nums) {
        int l = 0, r = nums.length;
        while(l<r){
            int m = l+(r-l)/2;
            if(nums[m]!=m){
                r = m;
            }else if(nums[m]==m){
                l=m+1;
            }
        }
        return l;
    }
}
```

​	

