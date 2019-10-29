import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Date;

public class CourseFile implements Serializable {
    private String fileName;
    private Date dateOfLoading;
    private File file;

    public CourseFile(String fileName, File file){
        this.fileName = fileName;
        dateOfLoading = new Date();
        this.file = file;
    }

    public Date getDateOfLoading() { return dateOfLoading; }

    public File getFile() { return file; }

    public String getFileName() { return fileName; }
    public void setFileName(String fileName) { this.fileName = fileName; }

    @Override
    public boolean equals(Object obj) {
        if(this == obj)
            return true;
        if(obj == null)
            return false;
        if(obj.getClass() != getClass())
            return false;
        CourseFile other = (CourseFile) obj;
        if (fileName == null) {
            if (other.fileName != null)
                return false;
        } else if (!fileName.equals(other.fileName))
            return false;
        if (dateOfLoading == null) {
            if (other.dateOfLoading != null)
                return false;
        } else if (!dateOfLoading.equals(other.dateOfLoading))
            return false;
        if (file == null) {
            if (other.file != null)
                return false;
        } else if (!file.equals(other.file))
            return false;
        return true;
    }

    @Override
    public String toString() {
        return fileName + " " + dateOfLoading;
    }
}
