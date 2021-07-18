/**
 * Implements in-place quaternary heapsort.
 *
 * The class has worst-case time and space complexities of O(nlog(n)) and
 * O(n) respectively, obtained from the quaternaryHeapsort method (see method
 * javadocs for more info).
 */
public class QuaternaryHeapsort {

    /**
     * Sorts the input array, in-place, using a quaternary heap sort. Builds
     * a quaternary (max) heap in-place bottom-up and uses this to sort the
     * array in ascending order.
     *
     * Let n = size of the input array. Worst-case runtime of O(nlog(n))
     * because second for loop is longer in runtime (O(n) vs. O(n/4) = O(n) for
     * the first for loop). quaternaryHeap() and swap have O(log(n)) and O(1)
     * worst-case time complexities respectively (each respective method's
     * javadocs for more info). Thus, altogether this is O(nlog(n)).
     * Worst-case space complexity of O(n) since modifying input array of
     * size n.
     *
     * @param input to be sorted (modified in place)
     */
    public static <T extends Comparable<T>> void quaternaryHeapsort(T[] input) {
        int start = 0;
        int size = input.length;

        // Builds quaternary max heap from bottom-up given an input array
        for (int i = size - 1; i >= start; i--) {
            quaternaryDownheap(input, i, size);
        }

        // Performs heap sort on this array so that it is sorted in ascending
        // order at the end.
        for (int i = size - 1; i >= start; i--) {
            swap(input, start, i);
            quaternaryDownheap(input, start, i);
        }
    }

    /**
     * Performs a downheap from the element in the given position on the given
     * max heap array. A downheap should restore the heap order by swapping
     * downwards as necessary. The array should be modified in place. You
     * should only consider elements in the input array from index 0 to index
     *  (size - 1) as part of the heap (i.e. pretend the input array stops
     *  after the inputted size).
     *
     *  Let n = size of the input array. Worst-case runtime of O(log(n))
     *  assuming while loop runs all the way down to the leaf node starting
     *  from root node (height of quaternary heap). Everything inside the
     *  while loop is O(1) because only assignments, evaluations and if
     *  statements. All other methods used below also have O(1) worst-case
     *  runtime. Worst-case space complexity of O(n) since modifying input array
     *  of size n.
     *
     * @param input array representing a quaternary max heap
     * @param start position in the array to start the downheap from
     * @param size the size of the heap in the input array, starting from index
     * 0.
     */
    public static <T extends Comparable<T>> void quaternaryDownheap(T[] input,
            int start, int size) {
        // Repeats until max heap property is satisfied in the quaternary heap
        while (hasLeft(start, size) && start < size) {
            // Find the left child index and assign largest child index to this
            // (temporarily).
            int leftIndex = left(start);
            int largestChildIndex = leftIndex;

            // If node has a middle left child, compare the two to update the
            // largest child index.
            if (hasMiddleLeft(start, size)) {
                int middleLeftIndex = middleLeft(start);

                if (input[largestChildIndex].compareTo(input[middleLeftIndex]) < 0) {
                    largestChildIndex = middleLeftIndex;
                }
            }

            // If node has a middle right child, compare the two to update the
            // largest child index.
            if (hasMiddleRight(start, size)) {
                int middleRightIndex = middleRight(start);

                if (input[largestChildIndex].compareTo(input[middleRightIndex]) < 0) {
                    largestChildIndex = middleRightIndex;
                }
            }

            // If node has a right child, compare the two to update the largest
            // child index.
            if (hasRight(start, size)) {
                int rightIndex = right(start);

                if (input[largestChildIndex].compareTo(input[rightIndex]) < 0) {
                    largestChildIndex = rightIndex;
                }
            }

            // Swap node and child with largest value to satisfy max heap
            // property at the current level.
            if (input[largestChildIndex].compareTo(input[start]) > 0) {
                swap(input, start, largestChildIndex);
            }

            // Update the start index so we can repeat until reach bottom of
            // the heap.
            start = largestChildIndex;
        }
    }

    /**
     * Returns left child of given node. O(1) worst-case runtime and space
     * complexity because only return statement.
     *
     * @param j index of given node
     * @return left most child of node
     */
    private static int left(int j) {
        return 4*j + 1;
    }

    /**
     * Returns middle left child of given node. O(1) worst-case runtime and
     * space complexity because only return statement.
     *
     * @param j index of given node
     * @return middle left child of node
     */
    private static int middleLeft(int j) {
        return 4*j + 2;
    }

    /**
     * Returns middle right child of given node. O(1) worst-case runtime and
     * space because only return statement
     *
     * @param j index of given node
     * @return middle right child of node
     */
    private static int middleRight(int j) {
        return 4*j + 3;
    }

    /**
     * Returns right child of given node. O(1) worst-case runtime and space
     * complexity because only return statement
     *
     * @param j index of given node
     * @return left most child of node
     */
    private static int right(int j) {
        return 4*j + 4;
    }

    /**
     * Checks if the given node has a left child. O(1) worst-case runtime and
     * space complexity because only return statement and evaluation.
     *
     * @param j index of given node
     * @param size the size of the heap in the input array, starting from index
     * 0.
     * @return whether the node has a left child or not
     */
    private static boolean hasLeft(int j, int size) {
        return left(j) < size;
    }

    /**
     * Checks if the given node has a middle left child. O(1) worst-case runtime
     * and space complexity because only return statement and evaluation.
     *
     * @param j index of given node
     * @param size the size of the heap in the input array, starting from index
     * 0.
     * @return whether the node has a middle left child or not
     */
    private static boolean hasMiddleLeft(int j, int size) {
        return middleLeft(j) < size;
    }

    /**
     * Checks if the given node has a middle right child. O(1) worst-case
     * runtime and space complexity because only return statement and
     * evaluation.
     *
     * @param j index of given node
     * @param size the size of the heap in the input array, starting from index
     * 0.
     * @return whether the node has a middle right child or not
     */
    private static boolean hasMiddleRight(int j, int size) {
        return middleRight(j) < size;
    }

    /**
     * Checks if the given node has a right child. O(1) worst-case runtime and
     * space complexity because only return statement and evaluation.
     *
     * @param j index of given node
     * @param size the size of the heap in the input array, starting from index
     * 0.
     * @return whether the node has a right child or not
     */
    private static boolean hasRight(int j, int size) {
        return right(j) < size;
    }

    /**
     * Exchanges the entries at indices i and j of a given array. O(1)
     * worst-case runtime and space complexity since only accessing array
     * indices and assignments.
     *
     * @param input array representing a quaternary max heap
     * @param i index in the array
     * @param j different index in the array
     */
    private static <T> void swap(T[] input, int i, int j) {
        // Use temporary variable to help swap
        T temp = input[i];
        input[i] = input[j];
        input[j] = temp;
    }

}
