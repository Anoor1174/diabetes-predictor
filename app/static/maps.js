// Initial empty map so Google Maps loads without errors
function initMap() {
    // Create default London-centred map
    const map = new google.maps.Map(document.getElementById("map"), {
        center: { lat: 51.5074, lng: -0.1278 }, // London default
        zoom: 12
    });
}

// Called when the user clicks the button
function findHealthyFood() {

    // Check the browser supports geolocation
    if (!navigator.geolocation) {
        alert("Location services are not supported on this device.");
        return;
    }

    // Request user's current location
    navigator.geolocation.getCurrentPosition((position) => {
        // Store the user's coordinates
        const userLoc = {
            lat: position.coords.latitude,
            lng: position.coords.longitude
        };

        // Recentre the map on the user
        const map = new google.maps.Map(document.getElementById("map"), {
            zoom: 14,
            center: userLoc
        });

        // Create a Places API service client
        const service = new google.maps.places.PlacesService(map);

        // Build the Places search request
        const request = {
            location: userLoc,
            radius: 5000,
            query: "healthy food organic market fresh produce"
        };

        // Run text search and handle results
        service.textSearch(request, (results, status) => {
            // Abort if the search failed
            if (status !== google.maps.places.PlacesServiceStatus.OK) {
                alert("No healthy food locations found nearby.");
                return;
            }

            // Drop a marker for each result
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