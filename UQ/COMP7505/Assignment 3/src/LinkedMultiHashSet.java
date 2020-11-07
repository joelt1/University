import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * LinkedMultiHashSet is an implementation of a (@see MultiSet),
 * using a hashtable as the internal data structure, and with
 * predictable iteration order based on the insertion order of
 * elements.
 * 
 * Its iterator orders elements according to when the first occurrence
 * of the element was added. When the multiset contains multiple
 * instances of an element, those instances are consecutive in the
 * iteration order. If all occurrences of an element are removed, after
 * which that element is added to the multiset, the element will appear
 * at the end of the iteration.
 * 
 * The internal hashtable array should be doubled in size after an add
 * that would cause it to be at full capacity. The internal capacity
 * should never decrease.
 * 
 * Collision handling for elements with the same hashcode (i.e. with
 * hashCode()) should be done using linear probing, as described in
 * lectures.
 *
 * The class has worst-case time and space complexities of O(mn^2) and
 * O(n) respectively, obtained from the add(element, count) method if
 * m >> n (see method javadocs for more info).
 *
 * @param <T> type of elements in the set
 */
public class LinkedMultiHashSet<T> implements MultiSet<T>, Iterable<T> {
    ListNode<T>[] multiset;
    int totalSize = 0;
    int distinctSize = 0;
    int capacity;
    int startNodeIndex = 0;
    int lastNodeIndex = -1;

    /**
     * Instantiates a linked list node to add to the linked list. O(1)
     * worst-case runtime and memory complexity since only declaring 4
     * variables.
     *
     * @param <T> the type of the linked list node (based on element stored).
     */
    private static class ListNode<T> {
        T element;              // The element to be stored
        int count;
        ListNode<T> next;       // The next list node for the current node
        ListNode<T> prev;       // The previous list node for the current node
    }

    /**
     * Instantiates a new linked multi hash set using an array of type list
     * node objects. Worst-case runtime and space complexities are O(n)
     * because initialising and assigning array elements to null.
     *
     * @param initialCapacity of the multiset
     */
    public LinkedMultiHashSet(int initialCapacity) {
        capacity = initialCapacity;
        multiset = (ListNode<T>[]) new ListNode[capacity];
    }

    /**
     * Adds the element to the set. If an equal element is already in the set,
     * increases its occurrence count by 1.
     *
     * Let n = capacity. Since findIndex has O(n) and O(1) worst-case time and
     * space complexity respectively, and increaseMultisetCapacity has
     * worst-case O(n) time and space complexity respectively, in the worst
     * scenario, add takes O(n*n) = O(n^2) and O(n*1) = O(n) time and space
     * complexity. However, the amortised (average) cost of an add operation is
     * O(n^2)/n = O(n) and O(n)/n = O(1) in terms of time and space complexity.
     * All other assignments, evaluations and conditionals are O(1).
     *
     *
     * @param element to add
     * @require element != null
     */
    @Override
    public void add(T element) {
        // Doesn't contain element
        if (!contains(element)) {
            ListNode<T> newNode = new ListNode<>();
            newNode.element = element;
            newNode.count = 1;
            // Find valid index in multiset using compression on the element's
            // hash code.
            int validIndex = compress(element.hashCode());

            // If first element in multiset, then assign start node index and
            // null to node's previous link.
            if (distinctSize == 0) {
                startNodeIndex = validIndex;
                newNode.prev = null;
            } else {
                // Perform linear probing if index in temporary multiset
                // already filled.
                if (multiset[validIndex] != null) {
                    validIndex = linearProbe(multiset, validIndex);
                }

                newNode.prev = multiset[lastNodeIndex];
                multiset[lastNodeIndex].next = newNode;
            }

            // Next link for latest node must always be null
            newNode.next = null;
            multiset[validIndex] = newNode;
            lastNodeIndex = validIndex;
            distinctSize++;
        } else {
            // If element already in multiset, find index in
            // multiset and increase count.
            int elementIndex = findIndex(element);
            multiset[elementIndex].count++;
        }

        // Double the multiset size if it has reached maximum
        // capacity.
        if (distinctSize == capacity) {
            increaseMultisetCapacity();
        }

        totalSize++;
    }

    /**
     * Adds count to the number of occurrences of the element in set.
     *
     * Let m = count of occurrences to be removed from the
     * multiset and n = capacity. Note that m can be > n. Worst-case
     * time complexity of O(mn) since m increases of element's count
     * in the multiset and findIndex is O(n) (only the outermost else
     * statement in add(element) runs). Average-case is O(n) if m << n
     * and using amortised cost for add(element). Worst- and average-case
     * space complexities are both O(1) (only the outermost else
     * statement in add(element) runs).
     *
     * @param element to add
     * @require element != null && count >= 0
     */
    @Override
    public void add(T element, int count) {
        while (count > 0) {
            add(element);
            count--;
        }
    }

    /**
     * Checks if the element is in the set (at least once).
     *
     *
     * Since findIndex has O(n) and O(1) worst-case time and space
     * complexity respectively (n = capacity) and the rest of the
     * method is accessing array index and returning the count, the
     * time and space complexity of the whole method is O(n) and
     * O(1) respectively.
     *
     * @param element to check
     * @return true if the element is in the set, else false.
     */
    @Override
    public boolean contains(T element) {
        int index = findIndex(element);
        return index != -1;
    }

    /**
     * Returns the count of how many occurrences of the given elements
     * there are currently in the set.
     *
     * Let n = capacity. Since findIndex has O(n) and O(1) worst-case
     * time and space complexity respectively and the rest of the method
     * is accessing multiset index and returning the count, the time and
     * space complexity of the whole method is O(n) and O(1) respectively.
     *
     * @param element to check
     * @return the count of occurrences of element
     */
    @Override
    public int count(T element) {
        int index = findIndex(element);
        if (index != -1) {
            return multiset[index].count;
        }

        return 0;
    }

    /**
     * Removes a single occurrence of element from the set.
     *
     * Worst-case time and space complexities are O(n) and O(1)
     * since findIndex has these for its time and space complexities.
     * All the conditionals, evaluations and assignments are O(1).
     *
     * @param element to remove
     * @throws NoSuchElementException if the set doesn't currently
     * contain the given element
     * @require element != null
     */
    @Override
    public void remove(T element) throws NoSuchElementException {
        if (!contains(element)) {
            throw new NoSuchElementException("Set doesn't currently contain " +
                    "the given element");
        }

        int index = findIndex(element);
        ListNode<T> node = multiset[index];

        // Need to remove node from multiset and update link between nodes if
        // only one occurrence of the element associated with the node.
        if (node.count == 1) {
            if (node.prev == null && node.next != null) {
                // First element
                node.next.prev = null;
                startNodeIndex = findIndex(node.next.element);
            } else if (node.prev != null && node.next == null) {
                // Last element
                node.prev.next = null;
                lastNodeIndex = findIndex(node.prev.element);
            } else if (node.prev != null && node.next != null) {
                // Middle element
                node.prev.next = node.next;
                node.next.prev = node.prev;
            }

            // Common case - also accounts for case where both prev and next
            // are null (first and last element).
            multiset[index] = null;
            distinctSize--;
        } else {
            node.count--;
        }

        totalSize--;
    }

    /**
     * Removes several occurrences of the element from the set.
     *
     * Let m = count of occurrences to be removed from the
     * multiset and n = capacity. Note m << n. Worst-case time complexity
     * of O(mn) since m decreases of element's count in the multiset and
     * remove(element) is O(n). Average-case is O(n) if m << n. Therefore,
     * average-case space complexity is O(1) if m << n since remove(element)
     * is O(1). We assume that the count here is <= the actual stored count
     * within the node (so only the outermost else statement in remove(element)
     * runs).
     *
     * @param element to remove
     * @param count the number of occurrences of element to remove
     * @throws NoSuchElementException if the set contains less than
     * count occurrences of the given element
     * @require element != null && count >= 0
     */
    @Override
    public void remove(T element, int count) throws NoSuchElementException {
        if (this.count(element) < count) {
            throw new NoSuchElementException("Set contains less than count " +
                    "occurrences of the given element");
        }

        while (count > 0) {
            remove(element);
            count--;
        }
    }

    /**
     * Returns the total count of all elements in the multiset. Note that
     * duplicates of an element all contribute to the count here.
     * O(1) worst-case runtime and space complexity since only return
     * statement.
     *
     * @return total count of elements in the collection
     */
    @Override
    public int size() { return totalSize; }

    /**
     * Returns the maximum number of *distinct* elements the internal data
     * structure can contain before resizing. O(1) worst-case runtime and
     * space complexity since only return statement.
     *
     * @return capacity of internal array
     */
    @Override
    public int internalCapacity() { return capacity; }

    /**
     * Returns the number of distinct elements currently stored in the set.
     * O(1) worst-case runtime and space complexity since only return
     * statement.
     *
     * @return count of distinct elements in the set
     */
    @Override
    public int distinctCount() {
        return distinctSize;
    }

    /**
     * Iterator orders elements according to when the first occurrence of
     * the element was added. When the multiset contains multiple instances
     * of an element, those instances are consecutive in the iteration order.
     * If all occurrences of an element are removed, after which that element
     * is added to the multiset, the element will appear at the end of the
     * iteration.
     *
     * O(1) overall worst-case runtime and space complexities (see individual
     * methods' javadocs below).
     *
     * @returns an iterator over the elements in in order from rightmost to
     * leftmost.
     */
    @Override
    public Iterator<T> iterator() {
        Iterator<T> iter = new Iterator<T>() {
            // Shifts every time next() is called
            ListNode<T> node = multiset[startNodeIndex];
            T element;
            // Temporarily assign to -1 to avoid null pointer exception
            // (when node is null, can't call node.count)
            int count = -1;


            /**
             * Returns whether or not there is another element in the multiset
             * O(1) worst-case runtime and space complexity since evaluation and
             * return statements only.
             *
             * @returns true if the there is another element, false otherwise.
             */
            @Override
            public boolean hasNext() {
                return node != null;
            }

            /**
             * Returns the next element in the multiset provided another
             * element exists. O(1) worst-case runtime and space
             * complexity since only conditional, evaluation and return
             * statements.
             *
             * @return the next element in the multiset.
             */
            @Override
            public T next() {
                // Safe to assign count to node.count after hasNext()
                // is satisfied (helps avoid null pointer exception).
                if (count == -1) {
                    count = node.count;
                }

                element = node.element;
                count--;
                // Once have displayed all the occurrences of an element,
                // can move on to next element and its occurrences.
                if (count == 0) {
                    node = node.next;
                    if (node != null) {
                        count = node.count;
                    }
                }

                return element;
            }
        };

        return iter;
    }

    /**
     * Finds the index in multiset for the desired element. If there is
     * no such element in the multiset, returns -1 to represent an
     * invalid index.
     *
     * If n = capacity, worst-case time complexity of O(n)
     * since iterating through multiset (array). Worst-case space
     * complexity of O(1) since only assigning to integer variable i.
     *
     * @param element in the multiset
     * @return index location in multiset if element is found, otherwise
     * -1.
     */
    private int findIndex(T element) {
        for (int i = 0; i < capacity; i++) {
            if (multiset[i] != null && multiset[i].element == element) {
                return i;
            }
        }

        return -1;
    }

    /**
     * Increases (doubles) the multiset's capacity once it has reached
     * max capacity. Does this by retrieving all existing nodes,
     * re-compressing them after finding their new hash code and
     * adding them to a temporary (resized) multiset which will
     * become the new multiset. Also re-assigns the start and end node
     * indices for the start node and end node locations in multiset.
     *
     * If n = original capacity, average-case time complexity of O(n)
     * since iterating through a full array is O(n) and performing
     * linear probing within is expected to be O(1) -> O(n*1) = O(n).
     * Worst-case space complexity of O(2*n) = O(n) because initialising
     * and assigning new multiset array of size double the original
     * capacity.
     */
    private void increaseMultisetCapacity() {
        ListNode<T> node;
        T element;
        int validIndex;
        // New capacity = double original capacity (doubling strategy used)
        capacity = 2*capacity;
        ListNode<T>[] tempMultiset = (ListNode<T>[]) new ListNode[capacity];

        for (int i = 0; i < multiset.length; i++) {
            node = multiset[i];
            element = node.element;
            // Need to do this because compression using new capacity
            // can yield different index to before.
            validIndex = compress(element.hashCode());

            // Perform linear probing if index in temporary multiset
            // already filled.
            if (tempMultiset[validIndex] != null) {
                validIndex = linearProbe(tempMultiset, validIndex);
            }

            // Update start node and last node indices based on the
            // updated valid index.
            if (node.prev == null) {
                startNodeIndex = validIndex;
            }
            if (node.next == null) {
                lastNodeIndex = validIndex;
            }

            tempMultiset[validIndex] = node;
        }

        multiset = tempMultiset;
    }

    /**
     *  Performs compression on the result after applying hash code to
     *  an element i.e. h(x) = h2(h1(x)) where compression is the  function
     *  h2. Returns an integer in the interval [0, capacity - 1].
     *
     *  If n = capacity, worst-case time and space complexities are O(1)
     *  since only evaluation and return statement (assuming obtaining a
     *  hash code runs in O(1) time).
     *
     * @param hashCode an integer representing a hash code for an element
     * @return a valid index for the multiset array.
     */
    private int compress(int hashCode) {
        return hashCode % capacity;
    }

    /**
     * Handles collisions in the hashtable by placing the colliding element
     * in the next (circularly) available table cell (in multiset).
     * Colliding items lump together causing future collisions to cause a
     * longer sequence of probes.
     *
     * If n = capacity, average-case time complexity of O(1) as it is expected
     * that the first or second cell after the given index should be empty.
     * Worst-case space complexity of O(1) since only one variable being
     * assigned (and reassigned).
     *
     * @param multiset
     * @param index
     * @return
     */
    private int linearProbe(ListNode<T>[] multiset, int index) {
        // Keep increasing index (circularly) until next empty (null)
        // cell is found.
        while (multiset[index] != null) {
            index++;
            index = (index % capacity);
        }

        return index;
    }
}