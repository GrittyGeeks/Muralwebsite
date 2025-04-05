const officeList = [
    { "name" : "Sri Chitra Art Gallery",
      "location" : [8.5366105,76.9347057] 
    },
    { "name" : "Bhavageetham Art Collections",
    "location" : [8.5366105,76.9347057] 
    },
    { "name" : "Raja Ravi Varma Art Gallery Kilimanoor",
    "location" : [8.7675967,76.8303416] 
    },
    { "name" : "Madre De Deus Church (Vettucaud Church)",
        "location" : [8.4962231,76.8964876] 
        },
    { "name" : "Kerala Museum",
            "location" : [10.0379375,76.2387198] 
            },
     { "name" : "Hill Palace Museum",
                "location" : [9.9526027,76.2862547] 
                }
]

var map = L.map('map') 
    .setView([10.8505, 76.2711], 7);  
 
    const accessToken = "your.mapbox.access.token";

    L.tileLayer("https://api.mapbox.com/styles/v1/mapbox/streets-v11/tiles/{z}/{x}/{y}?access_token=" + accessToken, {
      attribution: "Map data &copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors, " + 
        "<a href='https://www.mapbox.com/about/maps/'>Imagery © Mapbox</a>",
      maxZoom: 19,
    }).addTo(map);


L.tileLayer(
    'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
    ).addTo(map);



officeList.map(office => {
   
L.marker(office.location).addTo(map) 

    .bindPopup(office.name) 
})



window.addEventListener('scroll', function () {
    var swipeText = document.querySelector('.swipe-up-text');
    var sectionPosition = swipeText.getBoundingClientRect().top;

    if (sectionPosition <= window.innerHeight * 0.75) {
      swipeText.classList.add('swipe-up-text');
    }
  });
