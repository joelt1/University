package assignment1;

import java.util.*;

/**
 * Program to find whether there exists (a possible empty) sequence of trade agreements between traders, such that
 * after those agreements are formed, a given trader can trade items of a given type.
 *
 * @author Joel Thomas.
 */
public class TradeFinder {
    /**
     * Returns true if and only if there exists (a possible empty) sequence of agreements between traders in the set T
     * that can be formed, such that after those agreements are formed, trader t can trade items of type g; otherwise
     * the algorithm returns false.
     *
     * @param traders A set of traders.
     * @param t A trader that is in the set of traders.
     * @param g A type of item that trader t is willing to trade.
     * @return True if trader t can eventually trade items of type g, false otherwise.
     *
     * @require The set of traders is non-null, and does not contain null traders. The
     *          given trader, t, is non-null and is included in the set of traders. The
     *          given item type, g, is non-null and is included in the set of types of
     *          items that trader t is willing to trade.
     * 
     * @ensure Returns true if and only if there exists a (possibly empty) sequence of
     *         agreements between traders in the given set of traders that can be formed,
     *         such that after those agreements are formed, trader t can trade items of
     *         type g; otherwise the algorithm should return false.
     * 
     *         The method should not modify the set of traders provided as input, nor
     *         should it modify trader t, or item type g.
     * 
     *         (See the assignment handout for details and examples.)
     */
    public static boolean canTrade(Set<Trader> traders, Trader t, ItemType g) {
        // Trivial case - trader t naturally produces item g.
        if (t.getProducedItem().equals(g)) {
            return true;
        }

        // Initialise graph, form vertices (traders) and form edges.
        // Edges between vertices are based on items that both traders are mutually willing to trade.
        Graph graph = new Graph(traders);

        // Solve problem using edge relaxation methodology derived from Bellman-Ford algorithm.
        // Relax each edge up to a total of (|G.V| - 1) times
        for (int i = 1; i <= traders.size() - 1; i++) {
            for (Graph.Edge edge : graph.getEdges()) {
                // relaxEdge early returns true iff trader t ends up being able to trade g after an edge relaxation.
                // Stops further unnecessary computation.
                if (relaxEdge(graph, t, g, edge)) {
                    return true;
                }
            }
        }

        return false;
    }

    /**
     * "Relaxes" a given edge in the graph by checking whether any trade agreements can be formed between the two
     * adjacent traders based on the items they are mutually willing to trade and the items each of them can currently
     * trade.
     *
     * @param graph A graph representing traders as vertices and edges as items that any two traders are mutually
     *              willing to trade.
     * @param t A trader that is in the set of traders.
     * @param g A type of item that trader t is willing to trade.
     * @param edge A graph edge that stores items two traders are mutually willing to trade.
     * @return True if trader t can eventually trade items of type g, false otherwise.
     */
    private static boolean relaxEdge(Graph graph, Trader t, ItemType g, Graph.Edge edge) {
        // Items that vertex w (representing trader t) can trade.
        HashSet<ItemType> w_canTrade = graph.getVertexCanTrade(graph.getTraderVertexMap(t));
        // Items that vertices u and v (belonging to the given edge) can trade.
        HashSet<ItemType> u_canTrade = graph.getVertexCanTrade(graph.getEdgeVertex1(edge));
        HashSet<ItemType> v_canTrade = graph.getVertexCanTrade(graph.getEdgeVertex2(edge));

        // Iterate over all possible item pairs to check for a trade agreement between the two traders.
        for (ItemType gi : graph.getEdgeWillingItems(edge)) {
            for (ItemType gj : graph.getEdgeWillingItems(edge)) {
                // Ignore same item trade agreements.
                // Then check whether traders represented by vertices u and v can trade gi and gj respectively so that
                // after the trade agreement, u and v can now trade gj and gi respectively (swapped).
                if (!gi.equals(gj) && u_canTrade.contains(gi) && v_canTrade.contains(gj)) {
                    u_canTrade.add(gj);
                    v_canTrade.add(gi);

                    // Early return to stop forming additional trade agreements if the latest trade agreement results
                    // in the original trader t being able to trade items of type g.
                    if (w_canTrade.contains(g)) {
                        return true;
                    }
                }
            }
        }

        return false;
    }
}
