import java.io.Serializable;

public class Order implements Serializable {
    private String description;
    private boolean status;

    public Order(String description){
        this.description = description;
        status = false;
    }

    public String getDescription() { return description; }

    public boolean getStatus() { return status; }

    public void setStatus(boolean status) { this.status = status; }

    @Override
    public boolean equals(Object obj) {
        if(this == obj)
            return true;
        if(obj == null)
            return false;
        if(obj.getClass() != getClass())
            return false;
        Order other = (Order) obj;
        if (description == null) {
            if (other.description != null)
                return false;
        } else if (!description.equals(other.description))
            return false;
        return status == other.status;
    }

    @Override
    public String toString() {
        return description + " " + status;
    }
}
