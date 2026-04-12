// Initial empty map so Google Maps loads without errors
function initMap() {
    const map = new google.maps.Map(document.getElementById("map"), {
        center: { lat: 51.5074, lng: -0.1278 }, // London default
        zoom: 12
    });
}

// Called when the user clicks the button
function findHealthyFood() {

    if (!navigator.geolocation) {
        alert("Location services are not supported on this device.");
        return;
    }

    navigator.geolocation.getCurrentPosition((position) => {
        const userLoc = {
            lat: position.coords.latitude,
            lng: position.coords.longitude
        };

        const map = new google.maps.Map(document.getElementById("map"), {
            zoom: 14,
            center: userLoc
        });

        const service = new google.maps.places.PlacesService(map);

        const request = {
            location: userLoc,
            radius: 5000,
            query: "healthy food organic market fresh produce"
        };

        service.textSearch(request, (results, status) => {
            if (status !== google.maps.places.PlacesServiceStatus.OK) {
                alert("No healthy food locations found nearby.");
                return;
            }

            results.forEach(place => {
                new google.maps.Marker({
                    position: place.geometry.location,
                    map: map,
                    title: place.name
                });
            });
        });
    });
}
