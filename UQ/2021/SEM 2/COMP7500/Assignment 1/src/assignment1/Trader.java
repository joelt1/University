package assignment1;

import java.util.*;

/**
 * A class representing a trader.
 * 
 * Each trader has one type of item that it produces, and a set of types of items that the
 * trader is willing to trade. A trader is always willing to trade the type of item that
 * they produce. More than one trader can produce the same type of item.
 * 
 * DO NOT MODIFY THIS FILE IN ANY WAY.
 */
public final class Trader {

    /* The name of the trader. */
    private final String name;
    /* The type of item that this trader produces. */
    private final ItemType producedItem;
    /* The set of types of items that the trader is willing to trade. */
    private final HashSet<ItemType> tradableItems;

    /*
     * class invariant: name != null && producedItem != null && tradableItems != null &&
     * !tradableItems.contains(null) && tradable.contains(producedItem)
     */

    /**
     * @require name != null && producedItem != null && tradableItems != null &&
     *          !tradableItems.contains(null) && tradable.contains(producedItem)
     * @ensure Creates a new trader with the given name, the given type of the item that
     *         this trader produces, and the given set of types of items that the trader
     *         is willing to trade.
     */
    public Trader(String name, ItemType producedItem, HashSet<ItemType> tradableItems) {
        if (name == null) {
            throw new IllegalArgumentException("The trader's name cannot be null.");
        }
        if (producedItem == null) {
            throw new IllegalArgumentException(
                    "The type of item that the trader produces cannot be null.");
        }
        if (tradableItems == null || tradableItems.contains(null)) {
            throw new IllegalArgumentException(
                    "The types of items that the trader is willing to trade cannot be null");
        }
        if (!tradableItems.contains(producedItem)) {
            throw new IllegalArgumentException(
                    "A trader is always willing to trade the type of item "
                            + " that they produce.");
        }
        this.name = name;
        this.producedItem = producedItem;
        this.tradableItems = tradableItems;
    }

    /**
     * @ensure Returns the name of the trader.
     */
    public String getName() {
        return name;
    }

    /**
     * @ensure Returns the type of item that this trader produces.
     */
    public ItemType getProducedItem() {
        return producedItem;
    }

    /**
     * @ensure Returns (a copy of) the set of types of items that the trader is willing to
     *         trade.
     */
    public Set<ItemType> getTradableItems() {
        return new HashSet<ItemType>(tradableItems);
    }

    /**
     * @ensure Returns true if and only if the trader is willing to trade items of the
     *         given type.
     */
    public boolean willingToTrade(ItemType item) {
        return tradableItems.contains(item);
    }

    @Override
    public String toString() {
        return name + ": (" + producedItem + ", " + tradableItems + ")";
    }

}
