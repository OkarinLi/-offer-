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

#### 54 二叉搜索树的第二大节点

首先复习一下二叉搜索树

- 所有的左子树小于根节点，所有的右子树大于根节点 

- 对二叉树进行中序遍历会得到一个递增序列。

这题要找第k大个节点，那么也就是要递减序列的第k个，应该按照右-根-左的顺序进行遍历，这样得到递减序列，找出第k个即为所得。

```Java
class Solution {
    int res,k;
    public int kthLargest(TreeNode root, int k) {
        this.k = k;
        dfs(root);
        return res;
    }
    private void dfs(TreeNode root){
        if(root==null||k==0) return; //注意这里的终止条件 k=0意味着已经找到第k大节点，后面的递归就没有意义了
        dfs(root.right);
        if(--k==0) res = root.val;
        dfs(root.left);
    }
}
```

#### 68-1 二叉搜索树的最近公共祖先

很有意思的一题，需要注意的是当遍历到二叉搜索树的一个节点时，只有以下三种情况：

- p、q都在当前节点左子树

- p、q都在当前节点右子树

- 当前节点为分叉节点（即pq分别在当前节点的左右子树，一个值大于当前节点，一个值小于当前节点）

```Java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        TreeNode node = root;
        while(node!=null){
            if(p.val>node.val&&q.val>node.val) node=node.right;
            else if(p.val<node.val&&q.val<node.val) node=node.left;
            else break;
        }
        return node;
    }
}
```

#### 68-2 二叉树的公共最近祖先

这题相比上一题把条件放宽到二叉树，失去了二叉搜索树的性质让查找变难，需要用递归来查找，但是总体的思路是一样的，即找到一个节点使得这个节点是pq之一或pq分别在此节点的左右子树。

```Java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root==null||root==p||root==q) return root; //递归到空节点返回空，如果是pq就返回pq
        TreeNode leftNode = lowestCommonAncestor(root.left,p,q);//递归查找左子树
        TreeNode rightNode = lowestCommonAncestor(root.right,p,q);//递归查找右子树
        if((leftNode!=null&&rightNode!=null)) return root; //两个都不为空意味着l为p，r为q或者l为q，r为p
        return leftNode==null?rightNode:leftNode; //执行到这一句意味着l和r其中一个为空，也就意味着公共节点是pq之一，那么返回非空的那个就行了
    }
}
```

#### 34 二叉树中和为某一值的路径

最最经典的回溯法，思路还是很清晰的。

```java
class Solution {
    private List<List<Integer>> res = new ArrayList<>();
    private List<Integer> tem = new ArrayList<>();
    public List<List<Integer>> pathSum(TreeNode root, int target) {
        recur(root,target);
        return res;
    }
    private void recur(TreeNode root, int target) {
        //递归到空节点return，也可在下面递归调用时判断是否为空再调用
        if(root==null) return;
        target-=root.val;
        //进if说明找到了一条满足条件的路径
        if(target==0&&root.left==null&&root.right==null){
            tem.add(root.val);
            res.add(new ArrayList(tem));
            tem.remove(tem.size()-1);
            return;
        }
        //没有进if则执行下面四行
        tem.add(root.val);
        recur(root.left,target);
        recur(root.right,target);
        tem.remove(tem.size()-1);
    }
}
```

#### 33 二叉搜索树的后序遍历序列

这题思路是有的，肯定是递归判断左右子树是否满足条件，借鉴前序中序序列生成二叉树的写法，用上下界控制递归，就不用搞什么分割数组的操作了。

```Java
class Solution {
    public boolean verifyPostorder(int[] postorder) {
        return recur(postorder,0,postorder.length-1);
    }
    private boolean recur(int[] postorder,int i, int j){ //i，j为数组上下界
        if(i>=j) return true;
        int left = i;
        while(postorder[left]<postorder[j]) left++;
        int right = left;
        while(postorder[right]>postorder[j]) right++;
        //right==j意味着i到left-1都小于root，left到j-1都大于root，满足二叉搜索树的条件
        return right==j&&recur(postorder,i ,left-1)&&recur(postorder,left,j-1);
    }
}
```

#### 26-树的子结构

这题思路有的，一个函数用来递归查找当前树A是否包含树B，外层再来一个函数递归树A的每一个节点。问题出在怎么确定树A是否包含树B（因为不是AB相等，例如[4,1,2]也算是包含[4,1]的。瞄了一眼题解的代码发现只要B==null时不管A是什么都算作是包含在A里就可以作为递归出口了，妙蛙！

```Java
class Solution {
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if(A==null||B==null) return false;
        return isSub(A,B)||isSubStructure(A.left,B)||isSubStructure(A.right,B);
    }
    private boolean isSub(TreeNode A, TreeNode B){
        if(B==null) return true;
        if(A==null||A.val!=B.val) return false;
        return isSub(A.left,B.left)&&isSub(A.right,B.right);
    }
}
```

今天做点简单的吧，给咱上一盘双指针。

#### 25 合并两个有序链表

```Java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode root = new ListNode(0);
        ListNode tem = root;
        while(l1!=null&&l2!=null){
            if(l1.val<=l2.val){
                tem.next = l1;
                l1=l1.next;
            }else{
                tem.next = l2;
                l2=l2.next;
            }
            tem = tem.next;
        }
        tem.next = l1==null?l2:l1;
        return root.next;
    }
}
```

唯一要说的是空头结点的使用，可以让添加变得容易，最后返回root.next就好。

#### 57-1 和为s的两个数字

简单的双指针应用，目前在面对有序数组的时候有两个思路：

- 双指针

- 二分查找

```Java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int i = 0, j = nums.length-1;
        while(i<j){
            if(nums[i]+nums[j]<target){
                i++;
            }else if(nums[i]+nums[j]>target){
                j--;
            }else{
                return new int[]{nums[i],nums[j]};
            }
        }
        return new int[0];
    }
}
```

#### 57-2 和为s的连续正数序列

有了上题的铺垫很容易想到双指针，另外原来这种类型的双指针就是滑动窗口啊。

此题最恶心的是用Java做要返回int数组，就离谱，时间都花在怎么转回数组上了。记录一下做法，但是除了做题正经人谁写这种代码啊。

```java
ArrayList<int[]> resArray = new ArrayList<>();
resArray.toArray(new int[resArray.size()][]);
```



```Java
class Solution {
    public int[][] findContinuousSequence(int target) {
        ArrayList<int[]> resArray = new ArrayList<>();
        int i = 1, j = 2;
        while(i<=target/2){
            int sum = ((i+j)*(j-i+1))/2;
            if(sum<target){
                j++;
            }else if(sum>target){
                i++;
            }else{
                int[] temp = new int[j-i+1];
                for(int k=i;k<=j;k++){
                    temp[k-i] = k;
                }
                resArray.add(temp);
                i++;
            }
        }
        return resArray.toArray(new int[resArray.size()][]);
    }
}
```

#### 22 链表中倒数第k个节点

太熟悉以至于不知道说啥了

```Java
class Solution {
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode l=head,r=head;
        for(int i=0;i<k-1;i++){
            r=r.next;
        }
        while(r.next!=null){
            l=l.next;
            r=r.next;
        }
        return l;
    }
}
```

#### 21 调整数组顺序使奇数位于偶数前面

想到了要双指针然后奇偶交换，但是思维陷入误区，没想到一个从前往后以后从后往前。

```Java
class Solution {
    public int[] exchange(int[] nums) {
        int l = 0, r = nums.length-1;
        while(l<r){
            if(nums[l]%2==1){
                l++;
                continue;
            }
            if(nums[r]%2==0){
                r--;
                continue;
            }
            //交换时l指向从左到右第一个偶数，r指向从右到左第一个奇数
            int temp = nums[l];
            nums[l] = nums[r];
            nums[r] = temp;
            
        }
        return nums;
    }
}
```

如果要两个指针都从第一个数开始，那么就要用快慢指针，快指针去找第一个奇数，慢指针去找第一个可以放奇数的位置。

```Java
class Solution {
    public int[] exchange(int[] nums) {
        int s = 0, f = 0;
        while(f<nums.length){
            if(nums[f]%2==1){
                int temp = nums[s];
                nums[s] = nums[f];
                nums[f] = temp;
                s++;
            }
            f++;
        }
        return nums;
    }
}
```

#### 52 链表的第一个公共节点

自己的想法是两次遍历，第一次找到两个链表的差值，第二次让长的节点先开始遍历，相遇的时候就是第一个公共节点。

```Java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if(headA==null||headB==null) return null;
        ListNode A = headA, B = headB;
        int count = 0;
        while(A.next!=null&&B.next!=null){
            A=A.next;
            B=B.next;
        }
        if(A.next==null){
            while(B!=A&&B!=null){
                B=B.next;
                count++;
            }
            if(B==null) return null;
            A=headA;
            B=headB;
            for(;count>0;count--){
                B=B.next;
            }
            while(A!=B){
                A=A.next;
                B=B.next;
            }
            return A;
        }
        if(B.next==null){
            while(A!=B&&A!=null){
                A=A.next;
                count++;
            }
            if(A==null) return null;
            A=headA;
            B=headB;
            for(;count>0;count--){
                A=A.next;
            }
            while(A!=B){
                A=A.next;
                B=B.next;
            }
            return B;
        }
        return null;
    }
}
```

结果看了题解给我整不会了，确实挺巧妙的但是关键还是太tm浪漫了。

[图解 双指针法，浪漫相遇 - 两个链表的第一个公共节点 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/solution/shuang-zhi-zhen-fa-lang-man-xiang-yu-by-ml-zimingm/)

```shell
你变成我，走过我走过的路。
我变成你，走过你走过的路。
然后我们便相遇了..
```

```Java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode A=headA, B=headB;
        while(A!=B){
            A=A!=null ? A.next : headB;
            B=B!=null ? B.next :headA;
        }
        return A;
    }
}
```

这题解写的，估计这辈子忘不掉了。今天的刷题到此为止吧。

#### 64 求1+2+...+n

莫名其妙的题，感觉是脑筋急转弯，用递归可以过，题解里的展开方法不知道有啥用，先不管了。哪个公司出这个题来面试在我这直接印象分就没了。

```Java
class Solution {
    public int sumNums(int n) {
        return n==0? 0 : sumNums(n-1) + n;
    }
}
```

#### 18 24 简单链表题 略过

#### 42 连续子数组的最大和

经典动态规划，明天开始好好做一下动态规划吧，这题算是个预热。

```Java
class Solution {
    public int maxSubArray(int[] nums) {
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        for(int i=1;i<nums.length;i++){
            //注意这里的转移方程 当dp[i-1]<=0时，意味着其对下一个位置起副作用，因此下一个位置的最大值就是nums[i]它本身
            //换言之，如果dp[i-1]>0，下一个位置的最大值就是dp[i-1]+nums[i]
            dp[i] = dp[i-1]<=0?nums[i]:dp[i-1]+nums[i];
        }
        int max =Integer.MIN_VALUE;
        for(int i=0;i<nums.length;i++){
            if(dp[i]>max) max = dp[i];
        }
        return max;
    }
}
```

按照惯例dp问题肯定可以优化空间复杂度的，比如题解里的做法是直接在nums数组上进行修改的，不过这种做法修改了原数组。

```Java
class Solution {
    public int maxSubArray(int[] nums) {
        int res = nums[0];
        for(int i = 1; i < nums.length; i++) {
            nums[i] += Math.max(nums[i - 1], 0);
            res = Math.max(res, nums[i]);
        }
        return res;
    }
}
```

#### 63 股票的最大利润

这题由于只能买卖一次，比较简单，不该分在medium吧。

```Java
class Solution {
    public int maxProfit(int[] prices) {
        int res = 0;
        int low_price = Integer.MAX_VALUE;
        for(int i=0;i<prices.length;i++){
            if(prices[i]<low_price) low_price = prices[i];
            if(prices[i]-low_price>res) res = prices[i]-low_price;
        }
        return res;
    }
}
```

重点看下可以多次买卖的下面这题。

#### leetcode-122 买卖股票的最佳时机（二）

这题理论上该用动态规划，但是可以投机取巧，或者好听点说叫贪心算法。

```Java
class Solution {
    public int maxProfit(int[] prices) {
        if(prices.length==0) return 0;
        int res = 0;
        for(int i=1;i<prices.length;i++){
            if(prices[i]>prices[i-1]){
                res+=prices[i]-prices[i-1];
            }
        }
        return res;
    }
}
```

每次当天价格比前一天贵的时候就可以算作前一天买入当天卖出，最后的结果是正确的。如果用动态规划来做的话麻烦的多，需要一个二维数组来表示当天持有或不持有股票时的利润最大值。

```Java
class Solution {
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[][] dp = new int[n][2];
        //dp[i][0]表示第i天没有持有股票时的最大利润
        //dp[i][1]表示第i天持有股票时的最大利润
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for (int i = 1; i < n; ++i) {
            //前一天就没持有，或者前一天持有但是当天卖掉了
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            //前一天就持有，或是前一天未持有当天买入
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
        }
        return dp[n - 1][0];
    }
}

```

看懂上面这个的话还可以优化一下空间复杂度，每天的状态只和前一天有关，所以只要两个变量存前一天就可以了。

```Java
class Solution {
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int dp0 = 0, dp1 = -prices[0];
        for (int i = 1; i < n; ++i) {
            dp0 = Math.max(dp0, dp1 + prices[i]);
            dp1 = Math.max(dp1, dp0 - prices[i]);
        }
        return dp0;
    }
}
```



#### 47 礼物的最大价值

终于可以秒杀这种比较简单的动态规划问题了，这题不用保留原数组，所以直接在原来的数组上修改了。

```Java
class Solution {
    public int maxValue(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
       for(int i=1;i<n;i++){
           grid[0][i] += grid[0][i-1];
       }
       for(int i=1;i<m;i++){
           grid[i][0] += grid[i-1][0]; 
       }
       for(int i=1;i<m;i++){
           for(int j=1;j<n;j++){
               grid[i][j] += Math.max(grid[i-1][j],grid[i][j-1]);
           }
       }
        return grid[m-1][n-1];
    }
}
```

#### 50 第一个只出现一次的字符

很自然想到用hashmap来做，统计并存储每个字符出现的次数，第二次遍历来找到只出现一次的字符。

```Java
class Solution {
    public char firstUniqChar(String s) {
         HashMap<Character, Integer> dic = new HashMap<>();
         for(int i=0;i<s.length();i++){
             char c = s.charAt(i);
             if(dic.containsKey(c)){
                 dic.put(c,dic.get(c)+1);
             }else{
                 dic.put(c,1);
             }
         }
         for(int i=0;i<s.length();i++){
             char c = s.charAt(i);
             if(dic.get(c)==1) return c;
         }
         return ' ';
    }
}
```

题解的做法有两个可以学习的地方：

- hashmap可以直接用boolean，不需要统计每个的次数，只要知道是否出现超过一次就行了。

- 直接把String转成char数组，遍历和取值都更方便一点。

```Java
class Solution {
    public char firstUniqChar(String s) {
        HashMap<Character, Boolean> dic = new HashMap<>();
        char[] sc = s.toCharArray();
        for(char c : sc)
            dic.put(c, !dic.containsKey(c));
        for(char c : sc)
            if(dic.get(c)) return c;
        return ' ';
    }
}
```

#### 39 数组中出现次数超过一半的数字

延续上题的思路肯定是哈希咯，统计每个数字出现的次数。理论上是O(n)的复杂度，但是执行并不快，为什么呢？

```Java
class Solution {
    public int majorityElement(int[] nums) {
        HashMap<Integer,Integer> dic = new HashMap<>();
        for(int num : nums){
            if(dic.containsKey(num)){
                dic.put(num,dic.get(num)+1);
            }else{
                dic.put(num,1);
            }
        }
        for(int num:nums){
            if(dic.get(num)>nums.length/2) return num;
        }
        return -1;
    }
}
```

因为要找的元素次数超过长度一般，所以可以不讲武德的排序后取中间位置的元素，一定是所找元素，但是不讲武德的代价是O(nlongn)。

```Java
class Solution {
    public int majorityElement(int[] nums) {
        Arrays.sort(nums);
        return nums[nums.length/2];
    }
}
```

题解中贼tm秀的Boyer-Moore 投票算法，O(n)时间复杂度，O(1)空间复杂度 ,理论上是这题的最优解？

```Java
class Solution {
    public int majorityElement(int[] nums) {
        int count = 0;
        Integer candidate = null;
        
        for (int num : nums) {
            if (count == 0) {
                candidate = num;
            }
            count += (num == candidate) ? 1 : -1;
        }
        return candidate;
    }
}
```



#### 15 二进制中1的个数

经典位运算，关键在于怎么取到每一位，对1进行移位加与操作应该是基操了吧？

```Java
public class Solution {
    public int hammingWeight(int n) {
        int res = 0;
        for (int i = 0; i < 32; i++) {
            if ((n & (1 << i)) != 0) {
                res++;
            }
        }
        return res;
    }
}
```

