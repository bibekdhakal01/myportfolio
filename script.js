// Global Variables
let currentProject = 0;
let totalProjects = 1; // Will be updated dynamically
let isAnimating = false;
let particles = [];
let canvas, ctx;

// Settings and Configuration
const portfolioSettings = {
    theme: localStorage.getItem('portfolio-theme') || 'cyber',
    particlesEnabled: localStorage.getItem('particles-enabled') !== 'false',
    animationsEnabled: localStorage.getItem('animations-enabled') !== 'false',
    autoplayCarousel: localStorage.getItem('autoplay-carousel') !== 'false',
    particleCount: parseInt(localStorage.getItem('particle-count')) || 100,
    carouselSpeed: parseInt(localStorage.getItem('carousel-speed')) || 5000
};

// GitHub Configuration
const GITHUB_USERNAME = 'sulav3690';
const GITHUB_API_URL = `https://api.github.com/users/${GITHUB_USERNAME}/repos`;

// Project Data (Dynamic Management)
let projectData = [
    {
        id: 1,
        title: "Loading Projects...",
        description: "Fetching repositories from GitHub. Please wait...",
        image: "/lovable-uploads/e9a8805c-d083-466a-8687-78145015de97.png",
        tags: ["Loading"],
        github: `https://github.com/${GITHUB_USERNAME}`,
        demo: "",
        featured: true
    }
];

// Fetch GitHub Repositories
async function fetchGitHubRepos() {
    try {
        const response = await fetch(GITHUB_API_URL + '?sort=updated&per_page=10');
        if (!response.ok) {
            throw new Error('Failed to fetch repositories');
        }
        const repos = await response.json();
        
        // Filter out forked repos and convert to project format
        const projects = repos
            .filter(repo => !repo.fork && !repo.private)
            .slice(0, 6) // Limit to 6 projects
            .map((repo, index) => {
                // Extract language tags
                const tags = [];
                if (repo.language) tags.push(repo.language);
                if (repo.topics && repo.topics.length > 0) {
                    tags.push(...repo.topics.slice(0, 3));
                }
                if (tags.length === 0) tags.push('GitHub');
                
                return {
                    id: index + 1,
                    title: repo.name.replace(/-|_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                    description: repo.description || 'A GitHub repository showcasing development skills and project implementation.',
                    image: "/lovable-uploads/e9a8805c-d083-466a-8687-78145015de97.png",
                    tags: tags.slice(0, 5),
                    github: repo.html_url,
                    demo: repo.homepage || "",
                    featured: index < 2,
                    stars: repo.stargazers_count,
                    forks: repo.forks_count
                };
            });
        
        if (projects.length > 0) {
            projectData = projects;
            // Update the carousel after fetching
            updateCarouselWithNewProjects();
        } else {
            // No projects found, keep default placeholder
            console.log('No public repositories found, using default projects');
        }
    } catch (error) {
        console.error('Error fetching GitHub repositories:', error);
        // On error, keep the default placeholder project
        // This provides a better fallback than showing a loading message indefinitely
        projectData = [
            {
                id: 1,
                title: "GitHub Projects",
                description: "Visit my GitHub profile to see my latest projects and contributions. I'm continuously working on new and exciting projects!",
                image: "/lovable-uploads/e9a8805c-d083-466a-8687-78145015de97.png",
                tags: ["GitHub", "Open Source"],
                github: `https://github.com/${GITHUB_USERNAME}`,
                demo: "",
                featured: true
            }
        ];
        updateCarouselWithNewProjects();
    }
}

// Update carousel with fetched projects
function updateCarouselWithNewProjects() {
    const carouselContainer = document.querySelector('.carousel-container');
    if (!carouselContainer) return;
    
    // Update total projects count
    totalProjects = projectData.length;
    
    // Clear existing slides
    const existingSlides = carouselContainer.querySelectorAll('.project-slide');
    existingSlides.forEach(slide => slide.remove());
    
    // Create new slides from fetched data
    projectData.forEach((project, index) => {
        const slide = createProjectSlide(project, index);
        carouselContainer.appendChild(slide);
    });
    
    // Update dots
    updateCarouselDots();
    
    // Reset current project to 0
    currentProject = 0;
    
    // Update display
    updateProjectDisplay();
    updateProjectCounter();
}

// Create a project slide element
function createProjectSlide(project, index) {
    const slide = document.createElement('div');
    slide.className = `project-slide ${index === 0 ? 'active' : ''}`;
    slide.setAttribute('data-project', index);
    
    // Create project content container
    const projectContent = document.createElement('div');
    projectContent.className = 'project-content';
    
    // Create project image container
    const imageContainer = document.createElement('div');
    imageContainer.className = 'project-image';
    
    const projectGlow = document.createElement('div');
    projectGlow.className = 'project-glow';
    imageContainer.appendChild(projectGlow);
    
    const img = document.createElement('img');
    img.src = project.image || '/lovable-uploads/e9a8805c-d083-466a-8687-78145015de97.png';
    img.alt = escapeHtml(project.title);
    imageContainer.appendChild(img);
    
    if (project.featured) {
        const featuredBadge = document.createElement('div');
        featuredBadge.className = 'featured-badge';
        featuredBadge.textContent = 'Featured';
        imageContainer.appendChild(featuredBadge);
    }
    
    projectContent.appendChild(imageContainer);
    
    // Create project details container
    const detailsContainer = document.createElement('div');
    detailsContainer.className = 'project-details';
    
    const title = document.createElement('h3');
    title.className = 'project-title text-gradient-accent';
    title.textContent = project.title;
    detailsContainer.appendChild(title);
    
    const description = document.createElement('p');
    description.className = 'project-description';
    description.textContent = project.description;
    detailsContainer.appendChild(description);
    
    // Add stats if available
    if (project.stars !== undefined) {
        const statsDiv = document.createElement('div');
        statsDiv.className = 'project-stats';
        statsDiv.innerHTML = `
            <span><i class="fas fa-star"></i> ${parseInt(project.stars) || 0}</span>
            <span><i class="fas fa-code-branch"></i> ${parseInt(project.forks) || 0}</span>
        `;
        detailsContainer.appendChild(statsDiv);
    }
    
    // Create tags container
    const tagsContainer = document.createElement('div');
    tagsContainer.className = 'project-tags';
    project.tags.forEach(tag => {
        const tagSpan = document.createElement('span');
        tagSpan.className = 'tag';
        tagSpan.textContent = escapeHtml(tag);
        tagsContainer.appendChild(tagSpan);
    });
    detailsContainer.appendChild(tagsContainer);
    
    // Create buttons container
    const buttonsContainer = document.createElement('div');
    buttonsContainer.className = 'project-buttons';
    
    // GitHub button
    const githubButton = document.createElement('button');
    githubButton.className = 'btn btn-neon';
    githubButton.innerHTML = '<i class="fab fa-github"></i> View Code';
    githubButton.addEventListener('click', () => {
        if (isValidUrl(project.github)) {
            window.open(project.github, '_blank');
        }
    });
    buttonsContainer.appendChild(githubButton);
    
    // Demo button (if available)
    if (project.demo && isValidUrl(project.demo)) {
        const demoButton = document.createElement('button');
        demoButton.className = 'btn btn-outline';
        demoButton.innerHTML = '<i class="fas fa-external-link-alt"></i> Live Demo';
        demoButton.addEventListener('click', () => {
            window.open(project.demo, '_blank');
        });
        buttonsContainer.appendChild(demoButton);
    }
    
    detailsContainer.appendChild(buttonsContainer);
    projectContent.appendChild(detailsContainer);
    slide.appendChild(projectContent);
    
    return slide;
}

// Helper function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Helper function to validate URLs
function isValidUrl(url) {
    try {
        const parsedUrl = new URL(url);
        return parsedUrl.protocol === 'http:' || parsedUrl.protocol === 'https:';
    } catch {
        return false;
    }
}

// Update carousel dots
function updateCarouselDots() {
    const dotsContainer = document.querySelector('.carousel-dots');
    if (!dotsContainer) return;
    
    // Clear existing dots
    dotsContainer.innerHTML = '';
    
    // Create new dots
    projectData.forEach((_, index) => {
        const dot = document.createElement('div');
        dot.className = `dot ${index === 0 ? 'active' : ''}`;
        dot.setAttribute('onclick', `goToProject(${index})`);
        dotsContainer.appendChild(dot);
    });
}

// Theme Colors
const themes = {
    cyber: {
        primary: '#3b82f6',
        accent: '#c084fc',
        background: '#0f0f23',
        card: '#1a1a3e'
    },
    neon: {
        primary: '#00ff88',
        accent: '#ff0080',
        background: '#0a0a0a',
        card: '#1a1a1a'
    },
    matrix: {
        primary: '#00ff00',
        accent: '#ffff00',
        background: '#000000',
        card: '#001100'
    },
    synthwave: {
        primary: '#ff006b',
        accent: '#8a2be2',
        background: '#1a0b2e',
        card: '#16213e'
    }
};

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initSettings();
    initParticles();
    initNavigation();
    fetchGitHubRepos(); // Fetch GitHub repositories dynamically
    initProjectCarousel();
    initSkillAnimations();
    initScrollAnimations();
    initFormHandling();
    initEnhancedFormValidation();
    updateCurrentYear();
    initScrollIndicator();
    initThemeSystem();
    initSettingsPanel();
    initTouchSupport();
    initKeyboardShortcuts();
    initPerformanceMonitor();
});

// Particles Background System
function initParticles() {
    canvas = document.getElementById('particles-canvas');
    if (!canvas) return;
    
    ctx = canvas.getContext('2d');
    
    // Set canvas size
    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // Create particles
    function createParticles() {
        particles = [];
        const maxParticles = Math.min(portfolioSettings.particleCount, Math.floor((canvas.width * canvas.height) / 8000));
        const particleCount = portfolioSettings.particlesEnabled ? maxParticles : 0;
        
        for (let i = 0; i < particleCount; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                dx: (Math.random() - 0.5) * 0.5,
                dy: (Math.random() - 0.5) * 0.5,
                size: Math.random() * 2 + 1,
                opacity: Math.random() * 0.5 + 0.1
            });
        }
    }
    
    // Draw particles and connections
    function drawParticles() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        particles.forEach((particle, index) => {
            // Update position
            particle.x += particle.dx;
            particle.y += particle.dy;
            
            // Boundary check
            if (particle.x < 0 || particle.x > canvas.width) particle.dx *= -1;
            if (particle.y < 0 || particle.y > canvas.height) particle.dy *= -1;
            
            // Draw particle
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(59, 130, 246, ${particle.opacity})`;
            ctx.fill();
            
            // Draw connections
            particles.slice(index + 1).forEach(otherParticle => {
                const distance = Math.sqrt(
                    Math.pow(particle.x - otherParticle.x, 2) +
                    Math.pow(particle.y - otherParticle.y, 2)
                );
                
                if (distance < 100) {
                    ctx.beginPath();
                    ctx.moveTo(particle.x, particle.y);
                    ctx.lineTo(otherParticle.x, otherParticle.y);
                    ctx.strokeStyle = `rgba(59, 130, 246, ${0.1 * (1 - distance / 100)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            });
        });
    }
    
    // Animation loop
    function animate() {
        drawParticles();
        requestAnimationFrame(animate);
    }
    
    createParticles();
    animate();
    
    // Recreate particles on resize
    window.addEventListener('resize', createParticles);
}

// Navigation System
function initNavigation() {
    const navbar = document.getElementById('navbar');
    
    // Handle scroll effects
    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });
}

// Smooth scrolling function
function scrollToSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        const offsetTop = element.offsetTop - 80; // Account for fixed navbar
        
        window.scrollTo({
            top: offsetTop,
            behavior: 'smooth'
        });
    }
    
    // Close mobile menu if open
    const mobileNav = document.getElementById('mobile-nav');
    if (mobileNav && !mobileNav.classList.contains('hidden')) {
        toggleMobileMenu();
    }
}

// Scroll to top function
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Mobile menu toggle
function toggleMobileMenu() {
    const mobileNav = document.getElementById('mobile-nav');
    const menuIcon = document.getElementById('menu-icon');
    const closeIcon = document.getElementById('close-icon');
    
    if (mobileNav.classList.contains('hidden')) {
        mobileNav.classList.remove('hidden');
        menuIcon.classList.add('hidden');
        closeIcon.classList.remove('hidden');
    } else {
        mobileNav.classList.add('hidden');
        menuIcon.classList.remove('hidden');
        closeIcon.classList.add('hidden');
    }
}

// Project Carousel System
function initProjectCarousel() {
    updateProjectDisplay();
    updateProjectCounter();
}

function goToProject(index) {
    if (isAnimating || index === currentProject) return;
    
    isAnimating = true;
    const slides = document.querySelectorAll('.project-slide');
    const dots = document.querySelectorAll('.dot');
    
    // Remove active class from current slide
    slides[currentProject].classList.remove('active');
    dots[currentProject].classList.remove('active');
    
    // Add active class to new slide
    setTimeout(() => {
        currentProject = index;
        slides[currentProject].classList.add('active');
        dots[currentProject].classList.add('active');
        updateProjectCounter();
        isAnimating = false;
    }, 250);
}

function nextProject() {
    const nextIndex = (currentProject + 1) % totalProjects;
    goToProject(nextIndex);
}

function previousProject() {
    const prevIndex = (currentProject - 1 + totalProjects) % totalProjects;
    goToProject(prevIndex);
}

function updateProjectDisplay() {
    const slides = document.querySelectorAll('.project-slide');
    const dots = document.querySelectorAll('.dot');
    
    slides.forEach((slide, index) => {
        if (index === currentProject) {
            slide.classList.add('active');
        } else {
            slide.classList.remove('active');
        }
    });
    
    dots.forEach((dot, index) => {
        if (index === currentProject) {
            dot.classList.add('active');
        } else {
            dot.classList.remove('active');
        }
    });
}

function updateProjectCounter() {
    const currentProjectElement = document.getElementById('current-project');
    const totalProjectsElement = document.getElementById('total-projects');
    
    if (currentProjectElement) {
        currentProjectElement.textContent = currentProject + 1;
    }
    if (totalProjectsElement) {
        totalProjectsElement.textContent = totalProjects;
    }
}

// Auto-advance carousel (optional)
let carouselInterval;

function startCarouselAutoPlay() {
    if (portfolioSettings.autoplayCarousel) {
        carouselInterval = setInterval(nextProject, portfolioSettings.carouselSpeed);
    }
}

function stopCarouselAutoPlay() {
    if (carouselInterval) {
        clearInterval(carouselInterval);
    }
}

// Start auto-play and pause on hover
document.addEventListener('DOMContentLoaded', function() {
    const carousel = document.querySelector('.project-carousel');
    if (carousel) {
        if (portfolioSettings.autoplayCarousel) {
            startCarouselAutoPlay();
        }
        
        carousel.addEventListener('mouseenter', stopCarouselAutoPlay);
        carousel.addEventListener('mouseleave', () => {
            if (portfolioSettings.autoplayCarousel) {
                startCarouselAutoPlay();
            }
        });
    }
});

// Skill Bar Animations
function initSkillAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateSkillBars(entry.target);
            }
        });
    }, { threshold: 0.3 });
    
    const skillsSection = document.getElementById('skills');
    if (skillsSection) {
        observer.observe(skillsSection);
    }
}

function animateSkillBars(section) {
    const skillBars = section.querySelectorAll('.skill-progress, .soft-skill-progress');
    
    skillBars.forEach((bar, index) => {
        const width = bar.getAttribute('data-width');
        if (width) {
            setTimeout(() => {
                bar.style.width = width + '%';
            }, index * 100);
        }
    });
}

// Scroll Animations
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });
    
    // Observe elements for scroll animations
    const animatedElements = document.querySelectorAll('.animate-fade-in-up');
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.8s ease-out, transform 0.8s ease-out';
        observer.observe(el);
    });
}

// Scroll Indicator
function initScrollIndicator() {
    const scrollIndicator = document.querySelector('.scroll-indicator');
    if (!scrollIndicator) return;
    
    window.addEventListener('scroll', function() {
        if (window.scrollY > 100) {
            scrollIndicator.style.opacity = '0';
        } else {
            scrollIndicator.style.opacity = '1';
        }
    });
    
    scrollIndicator.addEventListener('click', function() {
        scrollToSection('about');
    });
}

// Contact Form Handling
function initFormHandling() {
    const form = document.getElementById('contactForm');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }
}

async function handleFormSubmit(event) {
    event.preventDefault();
    
    const submitBtn = document.getElementById('submitBtn');
    const submitText = document.getElementById('submitText');
    const submitIcon = document.getElementById('submitIcon');
    const loadingSpinner = document.getElementById('loadingSpinner');
    
    // Show loading state
    submitBtn.disabled = true;
    submitText.textContent = 'Sending...';
    submitIcon.classList.add('hidden');
    loadingSpinner.classList.remove('hidden');
    
    // Get form data
    const formData = new FormData(event.target);
    const data = {
        name: formData.get('name'),
        email: formData.get('email'),
        subject: formData.get('subject'),
        message: formData.get('message')
    };
    
    try {
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Show success message
        showToast('Success!', 'Your message has been sent successfully. I\'ll get back to you soon!');
        
        // Reset form
        event.target.reset();
        
    } catch (error) {
        console.error('Form submission error:', error);
        showToast('Error!', 'There was an error sending your message. Please try again.', 'error');
    } finally {
        // Reset button state
        submitBtn.disabled = false;
        submitText.textContent = 'Send Message';
        submitIcon.classList.remove('hidden');
        loadingSpinner.classList.add('hidden');
    }
}

// Toast Notification System
function showToast(title, message, type = 'success') {
    const toast = document.getElementById('toast');
    const toastTitle = toast.querySelector('.toast-title');
    const toastMessage = toast.querySelector('.toast-message');
    const toastIcon = toast.querySelector('.toast-icon i');
    
    // Update content
    toastTitle.textContent = title;
    toastMessage.textContent = message;
    
    // Update styling based on type
    if (type === 'error') {
        toast.style.background = 'var(--destructive)';
        toastIcon.className = 'fas fa-exclamation-triangle';
    } else {
        toast.style.background = 'var(--success)';
        toastIcon.className = 'fas fa-check';
    }
    
    // Show toast
    toast.classList.add('show');
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        hideToast();
    }, 5000);
}

function hideToast() {
    const toast = document.getElementById('toast');
    toast.classList.remove('show');
}

// Update current year in footer
function updateCurrentYear() {
    const yearElement = document.getElementById('current-year');
    if (yearElement) {
        yearElement.textContent = new Date().getFullYear();
    }
}

// Keyboard Navigation
document.addEventListener('keydown', function(event) {
    // Project carousel keyboard navigation
    if (event.target.closest('.project-carousel')) {
        switch(event.key) {
            case 'ArrowLeft':
                event.preventDefault();
                previousProject();
                break;
            case 'ArrowRight':
                event.preventDefault();
                nextProject();
                break;
        }
    }
    
    // Escape key to close mobile menu
    if (event.key === 'Escape') {
        const mobileNav = document.getElementById('mobile-nav');
        if (mobileNav && !mobileNav.classList.contains('hidden')) {
            toggleMobileMenu();
        }
    }
});

// Smooth scrolling for anchor links
document.addEventListener('click', function(event) {
    const target = event.target.closest('a[href^="#"]');
    if (target) {
        event.preventDefault();
        const sectionId = target.getAttribute('href').substring(1);
        scrollToSection(sectionId);
    }
});

// Preload critical images
function preloadImages() {
    const criticalImages = [
        '/lovable-uploads/1b779239-ee7c-4fbc-ac9e-003be87aee68.png',
        '/lovable-uploads/e9a8805c-d083-466a-8687-78145015de97.png',
        '/lovable-uploads/6027eded-575a-472a-9a03-90e5b7466f70.png',
        '/lovable-uploads/3329e892-7a07-4334-a59c-a469e12e7dd2.png'
    ];
    
    criticalImages.forEach(src => {
        const img = new Image();
        img.src = src;
    });
}

// Initialize image preloading
document.addEventListener('DOMContentLoaded', preloadImages);

// Performance optimization: Lazy load non-critical content
function initLazyLoading() {
    const lazyElements = document.querySelectorAll('[data-lazy]');
    
    const lazyObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const element = entry.target;
                const src = element.getAttribute('data-lazy');
                
                if (element.tagName === 'IMG') {
                    element.src = src;
                } else {
                    element.style.backgroundImage = `url(${src})`;
                }
                
                element.removeAttribute('data-lazy');
                lazyObserver.unobserve(element);
            }
        });
    });
    
    lazyElements.forEach(el => lazyObserver.observe(el));
}

// Initialize lazy loading
document.addEventListener('DOMContentLoaded', initLazyLoading);

// ================== ENHANCED FEATURES ==================

// Settings Management
function initSettings() {
    // Apply saved settings
    applyTheme(portfolioSettings.theme);
    if (!portfolioSettings.animationsEnabled) {
        document.documentElement.style.setProperty('--transition-smooth', 'none');
        document.documentElement.style.setProperty('--transition-bounce', 'none');
    }
}

function saveSettings() {
    Object.keys(portfolioSettings).forEach(key => {
        localStorage.setItem(`portfolio-${key.replace(/([A-Z])/g, '-$1').toLowerCase()}`, portfolioSettings[key]);
    });
}

// Theme System
function initThemeSystem() {
    const savedTheme = localStorage.getItem('portfolio-theme');
    if (savedTheme && themes[savedTheme]) {
        applyTheme(savedTheme);
    }
}

function applyTheme(themeName) {
    if (!themes[themeName]) return;
    
    const theme = themes[themeName];
    const root = document.documentElement;
    
    root.style.setProperty('--primary', theme.primary);
    root.style.setProperty('--accent', theme.accent);
    root.style.setProperty('--background', theme.background);
    root.style.setProperty('--card', theme.card);
    
    portfolioSettings.theme = themeName;
    saveSettings();
    
    // Update theme selector if exists
    const themeSelector = document.getElementById('theme-selector');
    if (themeSelector) {
        themeSelector.value = themeName;
    }
}

// Settings Panel
function initSettingsPanel() {
    createSettingsPanel();
    
    // Settings toggle shortcut
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === ',') {
            e.preventDefault();
            toggleSettingsPanel();
        }
    });
}

function createSettingsPanel() {
    const settingsHTML = `
        <div id="settings-panel" class="settings-panel glass-card hidden">
            <div class="settings-header">
                <h3><i class="fas fa-cog"></i> Portfolio Settings</h3>
                <button id="close-settings" class="btn-close">&times;</button>
            </div>
            <div class="settings-content">
                <div class="setting-group">
                    <label>Theme</label>
                    <select id="theme-selector" class="setting-select">
                        <option value="cyber">Cyber Blue</option>
                        <option value="neon">Neon Green</option>
                        <option value="matrix">Matrix</option>
                        <option value="synthwave">Synthwave</option>
                    </select>
                </div>
                
                <div class="setting-group">
                    <label>
                        <input type="checkbox" id="particles-toggle" ${portfolioSettings.particlesEnabled ? 'checked' : ''}> 
                        Enable Particles
                    </label>
                </div>
                
                <div class="setting-group">
                    <label>
                        <input type="checkbox" id="animations-toggle" ${portfolioSettings.animationsEnabled ? 'checked' : ''}> 
                        Enable Animations
                    </label>
                </div>
                
                <div class="setting-group">
                    <label>
                        <input type="checkbox" id="autoplay-toggle" ${portfolioSettings.autoplayCarousel ? 'checked' : ''}> 
                        Autoplay Carousel
                    </label>
                </div>
                
                <div class="setting-group">
                    <label>Particle Count: <span id="particle-count-value">${portfolioSettings.particleCount}</span></label>
                    <input type="range" id="particle-count" min="20" max="200" value="${portfolioSettings.particleCount}">
                </div>
                
                <div class="setting-group">
                    <label>Carousel Speed: <span id="carousel-speed-value">${portfolioSettings.carouselSpeed}ms</span></label>
                    <input type="range" id="carousel-speed" min="2000" max="10000" step="500" value="${portfolioSettings.carouselSpeed}">
                </div>
                
                <div class="setting-actions">
                    <button id="reset-settings" class="btn btn-outline">Reset to Default</button>
                    <button id="export-settings" class="btn btn-neon">Export Settings</button>
                </div>
            </div>
        </div>
        
        <button id="settings-toggle" class="settings-toggle" title="Settings (Ctrl+,)">
            <i class="fas fa-cog"></i>
        </button>
    `;
    
    document.body.insertAdjacentHTML('beforeend', settingsHTML);
    
    // Event listeners
    document.getElementById('settings-toggle').addEventListener('click', toggleSettingsPanel);
    document.getElementById('close-settings').addEventListener('click', toggleSettingsPanel);
    document.getElementById('theme-selector').addEventListener('change', (e) => applyTheme(e.target.value));
    
    document.getElementById('particles-toggle').addEventListener('change', (e) => {
        portfolioSettings.particlesEnabled = e.target.checked;
        saveSettings();
        if (e.target.checked) {
            initParticles();
        } else {
            clearParticles();
        }
    });
    
    document.getElementById('animations-toggle').addEventListener('change', (e) => {
        portfolioSettings.animationsEnabled = e.target.checked;
        saveSettings();
        document.documentElement.style.setProperty('--transition-smooth', 
            e.target.checked ? 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)' : 'none');
    });
    
    document.getElementById('autoplay-toggle').addEventListener('change', (e) => {
        portfolioSettings.autoplayCarousel = e.target.checked;
        saveSettings();
        if (e.target.checked) {
            startCarouselAutoPlay();
        } else {
            stopCarouselAutoPlay();
        }
    });
    
    document.getElementById('particle-count').addEventListener('input', (e) => {
        portfolioSettings.particleCount = parseInt(e.target.value);
        document.getElementById('particle-count-value').textContent = e.target.value;
        saveSettings();
        if (portfolioSettings.particlesEnabled) {
            initParticles();
        }
    });
    
    document.getElementById('carousel-speed').addEventListener('input', (e) => {
        portfolioSettings.carouselSpeed = parseInt(e.target.value);
        document.getElementById('carousel-speed-value').textContent = e.target.value + 'ms';
        saveSettings();
        if (portfolioSettings.autoplayCarousel) {
            stopCarouselAutoPlay();
            startCarouselAutoPlay();
        }
    });
    
    document.getElementById('reset-settings').addEventListener('click', resetSettings);
    document.getElementById('export-settings').addEventListener('click', exportSettings);
    
    // Initialize theme selector
    document.getElementById('theme-selector').value = portfolioSettings.theme;
}

function toggleSettingsPanel() {
    const panel = document.getElementById('settings-panel');
    panel.classList.toggle('hidden');
    
    if (!panel.classList.contains('hidden')) {
        panel.style.animation = 'slideInRight 0.3s ease-out';
    }
}

function resetSettings() {
    const defaults = {
        theme: 'cyber',
        particlesEnabled: true,
        animationsEnabled: true,
        autoplayCarousel: true,
        particleCount: 100,
        carouselSpeed: 5000
    };
    
    Object.assign(portfolioSettings, defaults);
    saveSettings();
    location.reload(); // Reload to apply all changes
}

function exportSettings() {
    const settingsBlob = new Blob([JSON.stringify(portfolioSettings, null, 2)], {
        type: 'application/json'
    });
    const url = URL.createObjectURL(settingsBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'portfolio-settings.json';
    a.click();
    URL.revokeObjectURL(url);
}

// Enhanced Particle System
function clearParticles() {
    if (canvas && ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    particles = [];
}

// Touch Support for Mobile
function initTouchSupport() {
    let touchStartX = 0;
    let touchEndX = 0;
    
    const carousel = document.querySelector('.project-carousel');
    if (!carousel) return;
    
    carousel.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
    });
    
    carousel.addEventListener('touchend', (e) => {
        touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
    });
    
    function handleSwipe() {
        const swipeThreshold = 50;
        const diff = touchStartX - touchEndX;
        
        if (Math.abs(diff) > swipeThreshold) {
            if (diff > 0) {
                nextProject(); // Swipe left - next project
            } else {
                previousProject(); // Swipe right - previous project
            }
        }
    }
}

// Enhanced Keyboard Shortcuts
function initKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Only if not typing in input fields
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        
        switch(e.key) {
            case 'h':
                scrollToSection('hero');
                break;
            case 'a':
                scrollToSection('about');
                break;
            case 'p':
                scrollToSection('projects');
                break;
            case 's':
                scrollToSection('skills');
                break;
            case 'c':
                scrollToSection('contact');
                break;
            case 'ArrowLeft':
                if (e.target.closest('.project-carousel')) {
                    e.preventDefault();
                    previousProject();
                }
                break;
            case 'ArrowRight':
                if (e.target.closest('.project-carousel')) {
                    e.preventDefault();
                    nextProject();
                }
                break;
            case ' ':
                if (e.target.closest('.project-carousel')) {
                    e.preventDefault();
                    toggleCarouselAutoPlay();
                }
                break;
        }
    });
}

function toggleCarouselAutoPlay() {
    portfolioSettings.autoplayCarousel = !portfolioSettings.autoplayCarousel;
    saveSettings();
    
    if (portfolioSettings.autoplayCarousel) {
        startCarouselAutoPlay();
        showToast('Autoplay Enabled', 'Carousel will now auto-advance');
    } else {
        stopCarouselAutoPlay();
        showToast('Autoplay Disabled', 'Carousel autoplay is now paused');
    }
}

// Performance Monitor
function initPerformanceMonitor() {
    let fps = 0;
    let lastTime = performance.now();
    
    function measureFPS(currentTime) {
        fps = Math.round(1000 / (currentTime - lastTime));
        lastTime = currentTime;
        
        // If FPS drops below 30, suggest turning off particles
        if (fps < 30 && portfolioSettings.particlesEnabled) {
            console.warn('Low FPS detected. Consider disabling particles for better performance.');
        }
        
        requestAnimationFrame(measureFPS);
    }
    
    requestAnimationFrame(measureFPS);
}

// Dynamic Project Management
function addProject(projectData) {
    // Add to project data array
    projectData.push(projectData);
    
    // Regenerate carousel HTML
    updateProjectCarousel();
    
    // Update total projects count
    totalProjects = projectData.length;
    updateProjectCounter();
}

function updateProjectCarousel() {
    const carouselContainer = document.querySelector('.carousel-container');
    if (!carouselContainer) return;
    
    carouselContainer.innerHTML = projectData.map((project, index) => `
        <div class="project-slide ${index === 0 ? 'active' : ''}" data-project="${index}">
            <div class="project-content">
                <div class="project-image">
                    <div class="project-glow"></div>
                    <img src="${project.image}" alt="${project.title}">
                    ${project.featured ? '<div class="featured-badge">Featured</div>' : ''}
                </div>
                <div class="project-details">
                    <h3 class="project-title text-gradient-accent">${project.title}</h3>
                    <p class="project-description">${project.description}</p>
                    <div class="project-tags">
                        ${project.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                    </div>
                    <div class="project-buttons">
                        <button class="btn btn-neon" onclick="window.open('${project.github}', '_blank')">
                            <i class="fab fa-github"></i> View Code
                        </button>
                        ${project.demo ? `<button class="btn btn-outline" onclick="window.open('${project.demo}', '_blank')">
                            <i class="fas fa-external-link-alt"></i> Live Demo
                        </button>` : ''}
                    </div>
                </div>
            </div>
        </div>
    `).join('');
    
    // Update dots
    const dotsContainer = document.querySelector('.carousel-dots');
    if (dotsContainer) {
        dotsContainer.innerHTML = projectData.map((_, index) => 
            `<div class="dot ${index === 0 ? 'active' : ''}" onclick="goToProject(${index})"></div>`
        ).join('');
    }
}

// Enhanced Form Validation
function initEnhancedFormValidation() {
    const form = document.getElementById('contactForm');
    if (!form) return;
    
    const inputs = form.querySelectorAll('input, textarea');
    
    inputs.forEach(input => {
        input.addEventListener('blur', validateField);
        input.addEventListener('input', clearFieldError);
    });
}

function validateField(e) {
    const field = e.target;
    const value = field.value.trim();
    let isValid = true;
    let errorMessage = '';
    
    // Remove existing error
    clearFieldError(e);
    
    switch(field.type) {
        case 'email':
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(value)) {
                isValid = false;
                errorMessage = 'Please enter a valid email address';
            }
            break;
        case 'text':
            if (value.length < 2) {
                isValid = false;
                errorMessage = 'Please enter at least 2 characters';
            }
            break;
        default:
            if (value.length < 1) {
                isValid = false;
                errorMessage = 'This field is required';
            }
    }
    
    if (!isValid) {
        showFieldError(field, errorMessage);
    } else {
        showFieldSuccess(field);
    }
    
    return isValid;
}

function showFieldError(field, message) {
    field.classList.add('error');
    
    let errorDiv = field.parentNode.querySelector('.field-error');
    if (!errorDiv) {
        errorDiv = document.createElement('div');
        errorDiv.className = 'field-error';
        field.parentNode.appendChild(errorDiv);
    }
    errorDiv.textContent = message;
}

function showFieldSuccess(field) {
    field.classList.remove('error');
    field.classList.add('success');
}

function clearFieldError(e) {
    const field = e.target;
    field.classList.remove('error', 'success');
    
    const errorDiv = field.parentNode.querySelector('.field-error');
    if (errorDiv) {
        errorDiv.remove();
    }
}

// Console Art and Debug Info
console.log(`
%c╔══════════════════════════════════════════════════════════════╗
║  🚀 FUTURISTIC PORTFOLIO - ENHANCED VERSION                  ║
║  ════════════════════════════════════════════════════════════ ║
║  Built with: HTML5, CSS3, Vanilla JavaScript                ║
║  Features: Dynamic Themes, Settings Panel, Touch Support    ║
║  Author: Sulav Sharma                                        ║
║  Version: 2.0.0                                              ║
╚══════════════════════════════════════════════════════════════╝

%cKeyboard Shortcuts:
• H - Home section
• A - About section  
• P - Projects section
• S - Skills section
• C - Contact section
• Ctrl+, - Settings panel
• Space - Toggle carousel autoplay (when focused on carousel)
• ← → - Navigate projects

%cSettings are automatically saved to localStorage.
`, 
'color: #00ff88; font-family: monospace; font-size: 12px;',
'color: #3b82f6; font-size: 11px;',
'color: #c084fc; font-size: 10px;'
);

// Expose useful functions globally for debugging
window.portfolioDebug = {
    settings: portfolioSettings,
    themes,
    projectData,
    applyTheme,
    resetSettings,
    toggleSettingsPanel
};

// Add typing effect to hero title (optional enhancement)
function initTypingEffect() {
    const typingElement = document.querySelector('.hero-title .animate-float');
    if (!typingElement) return;
    
    const originalText = typingElement.textContent;
    typingElement.textContent = '';
    
    let index = 0;
    const typeSpeed = 150;
    
    function typeCharacter() {
        if (index < originalText.length) {
            typingElement.textContent += originalText.charAt(index);
            index++;
            setTimeout(typeCharacter, typeSpeed);
        }
    }
    
    // Start typing effect after a delay
    setTimeout(typeCharacter, 1000);
}

// Initialize typing effect (uncomment if desired)
// document.addEventListener('DOMContentLoaded', initTypingEffect);

// Handle browser back/forward navigation
window.addEventListener('popstate', function(event) {
    if (event.state && event.state.section) {
        scrollToSection(event.state.section);
    }
});

// Add error handling for failed API calls or network issues
window.addEventListener('error', function(event) {
    console.error('Global error:', event.error);
    // Could show user-friendly error message here
});

// Accessibility improvements
function initAccessibility() {
    // Add focus management for keyboard navigation
    const focusableElements = document.querySelectorAll(
        'a, button, input, textarea, select, [tabindex]:not([tabindex="-1"])'
    );
    
    focusableElements.forEach(element => {
        element.addEventListener('focus', function() {
            this.style.outline = '2px solid var(--primary)';
            this.style.outlineOffset = '2px';
        });
        
        element.addEventListener('blur', function() {
            this.style.outline = '';
            this.style.outlineOffset = '';
        });
    });
    
    // Add skip link functionality
    const skipLink = document.querySelector('.skip-link');
    if (skipLink) {
        skipLink.addEventListener('click', function(event) {
            event.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.focus();
                target.scrollIntoView();
            }
        });
    }
}

// Initialize accessibility features
document.addEventListener('DOMContentLoaded', initAccessibility);

// Service Worker registration for PWA capabilities (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js')
            .then(function(registration) {
                console.log('ServiceWorker registration successful');
            })
            .catch(function(error) {
                console.log('ServiceWorker registration failed');
            });
    });
}

// Analytics tracking (placeholder - replace with your analytics service)
function trackEvent(eventName, eventData = {}) {
    // Google Analytics 4 example:
    // gtag('event', eventName, eventData);
    
    // Or custom analytics:
    console.log('Event tracked:', eventName, eventData);
}

// Track important user interactions
document.addEventListener('DOMContentLoaded', function() {
    // Track project views
    const projectSlides = document.querySelectorAll('.project-slide');
    projectSlides.forEach((slide, index) => {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    trackEvent('project_viewed', { 
                        project_index: index,
                        project_name: slide.querySelector('.project-title')?.textContent 
                    });
                }
            });
        }, { threshold: 0.5 });
        
        observer.observe(slide);
    });
    
    // Track contact form interactions
    const contactForm = document.getElementById('contactForm');
    if (contactForm) {
        contactForm.addEventListener('submit', function() {
            trackEvent('contact_form_submitted');
        });
    }
    
    // Track external link clicks
    const externalLinks = document.querySelectorAll('a[target="_blank"]');
    externalLinks.forEach(link => {
        link.addEventListener('click', function() {
            trackEvent('external_link_clicked', {
                url: this.href,
                text: this.textContent
            });
        });
    });
});

// Console welcome message
console.log(
    '%c🚀 Welcome to Sulav Sharma\'s Portfolio! %c\n' +
    'Built with modern web technologies\n' +
    'Feel free to explore the code and reach out if you have any questions!',
    'color: #3b82f6; font-size: 16px; font-weight: bold;',
    'color: #9ca3af; font-size: 12px;'
);