// JavaScript for expandable details in drift report
function toggleDetails(id, arrowElem) {
    var x = document.getElementById(id);
    if (x.style.display === "none" || x.style.display === "") {
        x.style.display = "block";
        if (arrowElem) arrowElem.innerHTML = "&#9660;"; // ▼
    } else {
        x.style.display = "none";
        if (arrowElem) arrowElem.innerHTML = "&#9654;"; // ►
    }
}
