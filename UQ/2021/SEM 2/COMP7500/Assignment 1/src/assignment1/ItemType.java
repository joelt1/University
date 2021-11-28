package assignment1;

/**
 * An immutable representation of a type of item. Such a type has a unique identifier. Two
 * types of items are equal if they have the same identifier.
 *
 * DO NOT MODIFY THIS FILE IN ANY WAY.
 */
public final class ItemType {

    // the unique identifier of the type
    private final int identifier;

    /**
     * Creates a new item type with the given identifier.
     */
    public ItemType(int identifier) {
        this.identifier = identifier;
    }

    /**
     * Returns the identifier of the item type.
     */
    public int identifier() {
        return identifier;
    }

    @Override
    public String toString() {
        return "g" + identifier;
    }

    @Override
    public boolean equals(Object object) {
        if (!(object instanceof ItemType)) {
            return false;
        }
        ItemType other = (ItemType) object;
        return (this.identifier == other.identifier);
    }

    @Override
    public int hashCode() {
        return identifier;
    }
}
