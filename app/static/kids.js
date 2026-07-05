const FOODS = [
    { emoji: "🍎", name: "Apple", group: "fruit", type: "healthy", fact: "Apples are full of fibre that helps your tummy!" },
    { emoji: "🍌", name: "Banana", group: "fruit", type: "healthy", fact: "Bananas give you quick energy for playing!" },
    { emoji: "🥦", name: "Broccoli", group: "veg", type: "healthy", fact: "Broccoli has vitamins that help you grow strong!" },
    { emoji: "🥕", name: "Carrot", group: "veg", type: "healthy", fact: "Carrots are great for your eyes!" },
    { emoji: "🍗", name: "Chicken", group: "protein", type: "healthy", fact: "Chicken helps build strong muscles!" },
    { emoji: "🥚", name: "Eggs", group: "protein", type: "healthy", fact: "Eggs are packed with protein for energy!" },
    { emoji: "🍞", name: "Wholegrain Bread", group: "grain", type: "healthy", fact: "Wholegrains keep your energy steady all day!" },
    { emoji: "🍚", name: "Brown Rice", group: "grain", type: "healthy", fact: "Rice gives your body long-lasting fuel!" },
    { emoji: "🥛", name: "Milk", group: "dairy", type: "healthy", fact: "Milk helps build strong bones!" },
    { emoji: "🍦", name: "Yoghurt", group: "dairy", type: "healthy", fact: "Yoghurt is great for a happy tummy!" },
    { emoji: "🥤", name: "Fizzy Drink", group: "treat", type: "treat", fact: "Fizzy drinks are a sometimes treat — water is better for every day!" },
    { emoji: "🍬", name: "Sweets", group: "treat", type: "treat", fact: "Sweets are a fun treat sometimes, not an everyday food!" },
    { emoji: "🍟", name: "Chips", group: "treat", type: "treat", fact: "Chips are yummy sometimes! Veggies are better for every day." },
    { emoji: "🍩", name: "Donut", group: "treat", type: "treat", fact: "Donuts are a special treat, not for every day!" },
];

const REQUIRED_PLATE_SIZE = 6;
const REQUIRED_GROUPS = 4;

let score = 0;
let plateItems = [];
let groupsCovered = new Set();

function renderFoodGrid() {
    const grid = document.getElementById("foodGrid");
    grid.innerHTML = "";
    FOODS.forEach((food, idx) => {
        const btn = document.createElement("button");
        btn.className = "food-card";
        btn.innerHTML = `<span class="food-emoji">${food.emoji}</span><span class="food-name">${food.name}</span>`;
        btn.onclick = () => pickFood(idx);
        grid.appendChild(btn);
    });
}

function pickFood(idx) {
    const food = FOODS[idx];
    const feedback = document.getElementById("feedback");

    if (food.type === "healthy") {
        if (plateItems.length >= REQUIRED_PLATE_SIZE) return;
        plateItems.push(food.emoji);
        groupsCovered.add(food.group);
        score += 10;
        feedback.textContent = `${food.emoji} ${food.fact}`;
        feedback.classList.remove("treat");
    } else {
        feedback.textContent = `${food.emoji} ${food.fact}`;
        feedback.classList.add("treat");
    }

    updateScoreboard();
    renderPlate();

    if (plateItems.length >= REQUIRED_PLATE_SIZE && groupsCovered.size >= REQUIRED_GROUPS) {
        setTimeout(showBadge, 500);
    } else if (plateItems.length >= REQUIRED_PLATE_SIZE) {
        setTimeout(showBadge, 500); // still celebrate a full plate even with fewer groups
    }
}

function renderPlate() {
    const plate = document.getElementById("plate");
    if (plateItems.length === 0) {
        plate.innerHTML = '<div class="plate-hint" id="plateHint">Your plate is empty — start tapping foods below!</div>';
        return;
    }
    plate.innerHTML = plateItems.map(e => `<span>${e}</span>`).join("");
}

function updateScoreboard() {
    document.getElementById("score").textContent = score;
    document.getElementById("plateCount").textContent = `${plateItems.length} / ${REQUIRED_PLATE_SIZE}`;
}

function showBadge() {
    const modal = document.getElementById("badgeModal");
    const title = document.getElementById("badgeTitle");
    const text = document.getElementById("badgeText");
    const emoji = document.getElementById("badgeEmoji");

    if (groupsCovered.size >= REQUIRED_GROUPS) {
        emoji.textContent = "🏆";
        title.textContent = "Balanced Plate Badge!";
        text.textContent = `Amazing! You built a colourful plate with ${groupsCovered.size} different food groups and scored ${score} points!`;
    } else {
        emoji.textContent = "🌟";
        title.textContent = "Plate Complete!";
        text.textContent = `Great job filling your plate! You scored ${score} points. Try mixing in even more food groups next time for the big trophy!`;
    }

    modal.classList.remove("hidden");
}

function playAgain() {
    score = 0;
    plateItems = [];
    groupsCovered = new Set();
    document.getElementById("feedback").textContent = "";
    document.getElementById("badgeModal").classList.add("hidden");
    updateScoreboard();
    renderPlate();
}

renderFoodGrid();
updateScoreboard();
renderPlate();