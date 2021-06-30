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

四川归来，先刷树相关的题目吧。

#### 27 二叉树的镜像

同leetcode226翻转二叉树，做过但看到又有点懵，只能说常刷常新吧2333。

核心思路其实非常简单，递归交换二叉树的左右子树就可以了。

```Java
class Solution {
    public TreeNode mirrorTree(TreeNode root) {
        if(root==null) return null;
        mirrorTree(root.left);
        mirrorTree(root.right);
        TreeNode temp = null;
        temp = root.left;
        root.left = root.right;
        root.right = temp;
        return root;
    }
}
```



#### 28 对称的二叉树

最开始写出的代码是这样的：

```Java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if(root==null) return true;
        return isSymmetric(root.left)&&isSymmetric(root.right);
    }

}
```

通过了三分之一的用例，看了一下失败用例**[1,2,2,null,3,null,3]**，发现是对“对称”的定义理解有偏差，并不是说一棵二叉树左子树和右子树都是对称的它就是对称的。正确的定义如下：

- 两个对称节点L和R的val相等
- L的左子节点值等于R的右子节点值
- L的右子节点值等于R的左子节点值

那么可以写出如下的代码：

```Java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if(root==null) return true;
        return traversal(root.left,root.right);
    }
    private boolean traversal(TreeNode node1,TreeNode node2){
        if(node1==null && node2==null){
            return true;
        }
        //此行可简化
        if(node1==null&&node2!=null || node1!=null&&node2==null || node1.val!=node2.val) return false;
        return traversal(node1.left,node2.right)&&traversal(node1.right,node2.left);
    }
}
```

逻辑基本没问题，上面代码中标注的那行由于编译会先执行上一个if，所以可以简化一下不影响结果。

```Java
if(node1==null || node2==null || node1.val!=node2.val) return false;
```

#### 55-1 二叉树深度

经典且简单，每个学过数据结构的人都会这题吧哈哈哈。

```Java
class Solution {
    public int maxDepth(TreeNode root) {
        if(root==null) return 0;
        return Math.max(maxDepth(root.left),maxDepth(root.right))+1;
    }
}
```

#### 55-2 平衡二叉树

第一想法肯定是根据55-1计算深度的函数来做，但是这样访问每个节点要递归，计算深度还要递归，理论上会慢一点但是结果还是100%通过了。

```Java
class Solution {
    public boolean isBalanced(TreeNode root) {
        if(root==null) return true;
        return Math.abs(maxDepth(root.left)-maxDepth(root.right))<=1&&isBalanced(root.left)&&isBalanced(root.right);
    }
    private int maxDepth(TreeNode root) {
        if(root==null) return 0;
        return Math.max(maxDepth(root.left),maxDepth(root.right))+1;
    }
}
```

看题解还有后序遍历+剪枝的做法，留个坑以后有机会再补充吧。

#### 32-1 从上到下打印二叉树

其实就是二叉树经典算法层序遍历，放在medium有点过分。借助队列实现，此处可以注意一些java的知识

- LinkedList和Queue的关系

- 把LinkedList当队列使用时offer和poll方法

- ArrayList和int[]的区别与联系

有机会去看看Java容器的原码，此题代码没什么好讲了思路很清晰：

```Java
class Solution {
    public int[] levelOrder(TreeNode root) {
        if(root==null) return new int[0];
        Queue<TreeNode> q = new LinkedList<>();
        ArrayList<Integer> temp = new ArrayList<>();
        q.offer(root);
        while(!q.isEmpty()){
            TreeNode node = q.poll();
            temp.add(node.val);
            if(node.left!=null) q.offer(node.left);
            if(node.right!=null) q.offer(node.right);
        }
        int[] res = new int[temp.size()];
        for(int i=0;i<res.length;i++){
            res[i]=temp.get(i);
        }
        return res;
    }
}
```

#### 32-2 从上到下打印二叉树(二)

这题放easy纯粹是因为有上一题吧，添加一个限制条件要同层的节点放在一个数组里。解决方法是每次处理的时候在内部再写一个循环，一次处理一层的节点。

```Java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if(root==null) return res;
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while(!q.isEmpty()){
            List<Integer> temp = new ArrayList<>();
            int layerNum = q.size();
            for(int i=0;i<layerNum;i++){
                TreeNode node = q.poll();
                temp.add(node.val);
                if(node.left!=null) q.offer(node.left);
                if(node.right!=null) q.offer(node.right);
            }
            res.add(temp);
        }
        return res;
    }
}
```

#### 32-3 从上到下打印二叉树(三)

再新加要求每层根据奇偶层正序或倒序输出，看到这个要求本菜鸡肯定是只能想到把偶数层的倒序，从上题代码改改得到：

```Java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
            if(root==null) return res;
            Queue<TreeNode> q = new LinkedList<>();
            q.offer(root);
            int l = 0; // l为层数
            while(!q.isEmpty()){
                List<Integer> temp = new ArrayList<>();
                int layerNum = q.size();
                for(int i=0;i<layerNum;i++){
                    TreeNode node = q.poll();
                    temp.add(node.val);
                    if(node.left!=null) q.offer(node.left);
                    if(node.right!=null) q.offer(node.right);
                }
                l++;
                if(l%2==1) res.add(temp);
                else {
                    Collections.reverse(temp);
                    res.add(temp);
                }
            }
            return res;
    }
}
```

但是如果只是这样这题有啥意义呢？看题解发现一个使用双端队列的做法，可以学习一下Java中怎么把LinkedList当双端队列用。

```Java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
            if(root==null) return res;
            Queue<TreeNode> q = new LinkedList<>();
            q.offer(root);
            int l = 0; // l为层数
            while(!q.isEmpty()){
                LinkedList<Integer> temp = new LinkedList<>();//双端队列
                int layerNum = q.size();
                l++;
                for(int i=0;i<layerNum;i++){
                    TreeNode node = q.poll();
                    if(l%2==1){
                        temp.addLast(node.val);
                    }else{
                        temp.addFirst(node.val);
                    }
                    if(node.left!=null) q.offer(node.left);
                    if(node.right!=null) q.offer(node.right);
                }
                res.add(temp);
            }
            return res;
    }
}
```

