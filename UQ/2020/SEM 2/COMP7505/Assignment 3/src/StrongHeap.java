/**
 * Implements a strong heap check for a binary tree which is satisfied if the
 * tree is a complete binary tree and also satisfies the strong max heap
 * property.
 *
 * The class has worst-case time and space complexities of O(log(n)) and
 * O(n) respectively, obtained from the isStrongHeap method (see method
 * javadocs for more info).
 */
public class StrongHeap {
    /**
     * Determines whether the binary tree with the given root node is
     * a "strong binary heap", as described in the assignment task sheet.
     *
     * A strong binary heap is a binary tree which is:
     *  - a complete binary tree, AND
     *  - its values satisfy the strong heap property
     *
     * Let n = the number of nodes in the tree. 2*O(log(n)) = O(log(n))
     * worst-case time complexity because height of tree is log(n) and searching
     * from root to leaf node twice in both methods below. O(n) worst-case space
     * complexity because have n nodes, each of which take up O(1) space.
     *
     * @param root root of a binary tree, cannot be null
     * @return true if the tree is a strong heap, otherwise false
     */
    public static boolean isStrongHeap(BinaryTree<Integer> root) {
        return isCompleteCheck(root) && isStrongHeapCheck(root);
    }

    /**
     * Checks whether a binary tree is a complete binary tree. Starts from the
     * root node and progresses down the tree due to recursive calls.
     *
     * Let n = the number of nodes in the tree. O(log(n)) worst-case
     * time complexity because height of tree is log(n) and searching from
     * root to leaf node. Note that all the conditional statements below before
     * the final recursive return statement are O(1). O(n) worst-case space
     * complexity because have n nodes, each of which take up O(1) space.
     *
     * @param node node of a binary tree
     * @return true if the tree is a complete binary tree, false otherwise
     */
    private static boolean isCompleteCheck(BinaryTree<Integer> node) {
        if (node.isLeaf()) {
            // Trivial case at leaf node
            return true;
        } else if (hasLeft(node) && !hasRight(node)) {
            // Check further into the tree by progressing down left subtree
            return isCompleteCheck(node.getLeft());
        } else if (!hasLeft(node) && hasRight(node)) {
            // Fails check if don't have left node but have a right node
            return false;
        }

        // Perform entire check again on both left and right children
        return isCompleteCheck(node.getLeft()) && isCompleteCheck(node.getRight());
    }

    /**
     * Checks whether a binary tree satisfies the strong (max) heap property.
     * Starts from the root node and progresses down the tree due to recursive
     * calls.
     *
     * Let n = the number of nodes in the tree. O(log(n)) worst-case
     * time complexity because height of tree is log(n) and searching from
     * root to leaf node. Note that all the conditional statements below before
     * the final recursive return statement are O(1). O(n) worst-case space
     * complexity because have n nodes, each of which take up O(1) space.
     *
     * @param node node of a binary tree
     * @return true if the tree satisfies the strong (max) heap property,
     * false otherwise
     */
    private static boolean isStrongHeapCheck(BinaryTree<Integer> node) {
        // Trivial cases for null and leaf nodes
        if (node == null) {
            return true;
        } else if (node.isLeaf()) {
            return true;
        } else {
            // Check condition x < parent(x) on left and right children if they
            // exist
            if (hasLeft(node) && node.getLeft().getValue() >= node.getValue()) {
                return false;
            } else if (hasRight(node) && node.getRight().getValue() >= node.getValue()) {
                return false;
            }

            // Check condition x + parent(x) < parent(parent(x)) for left and
            // right children of left child of node if they all exist.
            if (hasLeft(node) && !node.getLeft().isLeaf()) {
                if (hasLeft(node.getLeft())) {
                    if (node.getLeft().getLeft().getValue() +
                            node.getLeft().getValue() >= node.getValue()) {
                        return false;
                    }
                } else if (hasRight(node.getLeft())) {
                    if (node.getLeft().getRight().getValue() +
                            node.getLeft().getValue() >= node.getValue()) {
                        return false;
                    }
                }
            }

            // Check condition x + parent(x) < parent(parent(x)) for left and
            // right children of right child of node if they all exist
            if (hasRight(node) && !node.getRight().isLeaf()) {
                if (hasLeft(node.getRight())) {
                    if (node.getRight().getLeft().getValue() +
                            node.getRight().getValue() >= node.getValue()) {
                        return false;
                    }
                } else if (hasRight(node.getRight())) {
                    if (node.getRight().getRight().getValue() +
                            node.getRight().getValue() >= node.getValue()) {
                        return false;
                    }
                }
            }
        }

        // Perform entire check again on both left and right children
        return isStrongHeapCheck(node.getLeft()) && isStrongHeapCheck(node.getRight());

    }

    /**
     * Checks if the given node has a left child. O(1) worst-case runtime and
     * space complexity because only return statement and evaluation.
     *
     * @param node node of a binary tree
     * @return whether the given node has a left child
     */
    private static boolean hasLeft(BinaryTree<Integer> node) {
        return node.getLeft() != null;
    }

    /**
     * Checks if the given node has a right child. O(1) worst-case runtime and
     * space complexity because only return statement and evaluation.
     *
     * @param node node of a binary tree
     * @return whether the given node has a right child
     */
    private static boolean hasRight(BinaryTree<Integer> node) {
        return node.getRight() != null;
    }
}
