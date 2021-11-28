package assignment1;

import java.util.*;

/**
 * Graph implementation consisting of vertices and edges for the given problem.
 *
 * @author Joel Thomas.
 */
public class Graph {
    private final ArrayList<Vertex> vertices;               // Stores all the graph vertices.
    private final ArrayList<Edge> edges;                    // Stores all the graph edges.
    private final HashMap<Trader, Vertex> traderVertexMap;  // HashMap that maps each trader to its vertex.

    /**
     * Implementation of graph vertices. Each vertex represents an individual trader, hence stores the trader as well
     * as a set of unique items the trader currently canTrade.
     */
    public class Vertex {
        private final Trader trader;                            // The trader representing the vertex.
        private HashSet<ItemType> canTrade = new HashSet<>();   // The items the trader can currently trade.

        private Vertex(Trader trader) {
            this.trader = trader;
            this.canTrade.add(trader.getProducedItem());
        }
    }

    /**
     * Implementation of graph edges. Each edge stores adjacent traders based on the items they are mutually willing
     * to trade.
     */
    public class Edge {
        private Vertex vertex1;                         // Vertex u in the edge (u, v).
        private Vertex vertex2;                         // Vertex v in the edge (u, v).
        private final HashSet<ItemType> willingItems;   // Items both traders are mutually willing to trade.

        private Edge(Vertex vertex1, Vertex vertex2, HashSet<ItemType> willingItems) {
            this.vertex1 = vertex1;
            this.vertex2 = vertex2;
            this.willingItems = willingItems;
        }
    }

    /**
     * Initialises the graph by creating vertices from the set of traders given as input and forming edges based on
     * the items any pair of traders are mutually willing to trade.
     *
     * @param traders A set of traders.
     */
    public Graph (Set<Trader> traders) {
        // Initialise private variables
        this.vertices = new ArrayList<>();
        this.edges = new ArrayList<>();
        this.traderVertexMap = new HashMap<>();

        // Create a single vertex for each trader in traders
        for (Trader trader : traders) {
            createVertex(trader);
        }

        // Create edges based on the items any pair of traders are mutually willing to trade.
        HashSet<ItemType> willingItems;
        for (int i = 0; i < vertices.size(); i++) {
            // Ensure traders aren't the same person
            for (int j = i + 1; j < vertices.size(); j++) {
                Vertex u = vertices.get(i);
                Vertex v = vertices.get(j);

                // Set intersection to get items both traders are mutually willing to trade.
                willingItems = new HashSet<>(u.trader.getTradableItems());
                willingItems.retainAll(v.trader.getTradableItems());

                // Need at least 2 or more items in the intersection.
                // A valid trade must always result in the exchange of two items.
                if (willingItems.size() > 1) {
                    createEdge(u, v, willingItems);
                }
            }
        }
    }

    /**
     * Helper method to create a vertex in the graph.
     *
     * @param trader A trader that is in the set of traders.
     */
    private void createVertex(Trader trader) {
        Vertex vertex = new Vertex(trader);
        // Add the resulting trader-vertex (key-value) pair to the map
        this.traderVertexMap.put(trader, vertex);
        this.vertices.add(vertex);
    }

    /**
     * Helper method to create an edge in the graph.
     *
     * @param vertex1 Vertex u in the edge (u, v).
     * @param vertex2 Vertex v in the edge (u, v).
     * @param willingItems Items both traders are mutually willing to trade.
     */
    private void createEdge(Vertex vertex1, Vertex vertex2, HashSet<ItemType> willingItems) {
        Edge edge = new Edge(vertex1, vertex2, willingItems);
        this.edges.add(edge);
    }

    /**
     * Getter method to retrieve the trader representing a given vertex.
     *
     * @param vertex A given vertex in the graph.
     * @return The resulting trader for that vertex.
     */
    public Trader getTrader(Vertex vertex) { return vertex.trader; }

    /**
     * Getter method to retrieve all the edges of the graph.
     *
     * @return All edges for the graph.
     */
    public ArrayList<Edge> getEdges() { return this.edges; }

    /**
     * Getter method to obtain the vertex represented by a given trader.
     *
     * @param t A trader that is in the set of traders.
     * @return The vertex represented by the trader.
     */
    public Vertex getTraderVertexMap(Trader t) { return this.traderVertexMap.get(t); }

    /**
     * Getter method to obtain the vertex u in the edge (u, v).
     *
     * @param edge An edge in the graph.
     * @return One of the adjacent vertices in the edge.
     */
    public Vertex getEdgeVertex1(Edge edge) { return edge.vertex1; }

    /**
     * Getter method to obtain the vertex v in the edge (u, v).
     *
     * @param edge An edge in the graph.
     * @return The other adjacent vertex in the edge.
     */
    public Vertex getEdgeVertex2(Edge edge) { return edge.vertex2; }

    /**
     * Getter method to obtain a set of items two traders are mutually willing to trade for a given edge.
     *
     * @param edge An edge in the graph.
     * @return A set of items both traders (adjacent vertices) are mutually willing to trade.
     */
    public HashSet<ItemType> getEdgeWillingItems(Edge edge) { return edge.willingItems; }

    /**
     * Getter method to obtain the set of items a given trader can currently trade.
     *
     * @param vertex A vertex in the graph.
     * @return A set of items the given trader (represented by the vertex) can currently trade.
     */
    public HashSet<ItemType> getVertexCanTrade(Vertex vertex) { return vertex.canTrade; }
}
