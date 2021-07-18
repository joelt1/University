import java.util.Comparator;

/**
 * A comparator for Binary Trees.
 *
 * The class has worst-case time and space complexities of O(log(n)) and O(n)
 * respectively, obtained from the compare method (see method javadocs for more
 * info).
 */
public class BinaryTreeComparator<E extends Comparable<E>> implements Comparator<BinaryTree<E>> {

    /**
     * Compares two binary trees with the given root nodes.
     * <p>
     * Two nodes are compared by their left childs, their values, then their
     * right childs, in that order. A null is less than a non-null, and equal
     * to another null.
     *
     * Let n = the number of nodes in the tree. 2*O(log(n)) = O(log(n))
     * worst-case time complexity because height of tree is log(n) and
     * in worst-case, obtain equality comparing left subtrees down to the
     * subtree's deepest leaf node, equality comparing values of the root node
     * and either equality / less than / greater than at the deepest right
     * subtree leaf node after comparing the right subtrees. O(n) worst-case
     * space complexity because have n nodes, each of which take up O(1) space.
     *
     * @param tree1 root of the first binary tree, may be null.
     * @param tree2 root of the second binary tree, may be null.
     * @return -1, 0, +1 if tree1 is less than, equal to, or greater than
     * tree2, respectively.
     */
    @Override
    public int compare(BinaryTree<E> tree1, BinaryTree<E> tree2) {
        // Altogether 4 base cases to consider - trees likely to be subtrees
        // due to recursive calls below.
        if (tree1 == null && tree2 == null) {
            // First binary tree equal to second binary tree
            return 0;
        } else if (tree1 == null && tree2 != null) {
            // First binary tree less than second binary tree
            return -1;
        } else if (tree1 != null && tree2 == null) {
            // First binary tree greater than second binary tree
            return 1;
        } else if (tree1.isLeaf() && tree2.isLeaf()) {
            // Compare values if both are lead nodes
            int result = tree1.getValue().compareTo(tree2.getValue());
            if (result < 0) {
                return -1;
            } else if (result > 0) {
                return 1;
            } else {
                return 0;
            }
        }

        // Compare result of left subtree analysis first
        int left_subtree_result = compare(tree1.getLeft(), tree2.getLeft());
        if (left_subtree_result == 0) {
            // If result is equal, then compare the values of the root node
            int value_result = tree1.getValue().compareTo(tree2.getValue());
            if (value_result < 0) {
                return -1;
            } else if (value_result > 0) {
                return 1;
            } else {
                // If still equal, use result of right subtree analysis last
                int right_subtree_result = compare(tree1.getRight(),
                        tree2.getRight());
                return right_subtree_result;
            }
        }

        // Returns in the event that left subtree analysis does not yield
        // equality.
        return left_subtree_result;
    }
}