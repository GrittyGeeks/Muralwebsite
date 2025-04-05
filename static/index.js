let navbar = document.querySelector('.navbar');
let menuBar = document.querySelector('#menu-bar');

menuBar.addEventListener('click', () => {
    navbar.classList.toggle('active');
});

document.addEventListener('click', (event) => {
    if (!navbar.contains(event.target) && !menuBar.contains(event.target)) {
        navbar.classList.remove('active');
    }
});





var swiper = new Swiper(".murals-row", {
    spaceBetween: 30,
    loop:true,
    centeredSlides:true,
    autoplay:{
        delay:9000,
        disableOnInteraction:false,
    },
    pagination: {
      el: ".swiper-pagination",
      clickable: true,
    },
    breakpoints: {
      0: {
        slidesPerView: 1,
      },
      768: {
        slidesPerView: 2,
      },
      1024: {
        slidesPerView: 3,
      },
    },
  });

  

