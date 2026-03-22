/* Forum Module for Morpheme */

const Forum = {
    categories: [],
    currentCategoryId: null,
    currentPostId: null,
    initialized: false,

    init: async function () {
        if (this.initialized) return;
        console.log("[Forum] Initializing forum module...");
        this.setupEventListeners();
        await this.loadCategories();
        this.initialized = true;
    },

    setupEventListeners: function () {
        // New post button
        const newPostBtn = document.getElementById('forum-new-post-btn');
        if (newPostBtn) {
            newPostBtn.addEventListener('click', () => this.showCreateView());
        }

        // Back to list button
        const backToListBtn = document.getElementById('forum-back-to-list');
        if (backToListBtn) {
            backToListBtn.addEventListener('click', () => this.showListView());
        }

        // Cancel create button
        const cancelCreateBtn = document.getElementById('forum-cancel-create');
        if (cancelCreateBtn) {
            cancelCreateBtn.addEventListener('click', () => this.showListView());
        }

        // Post create form
        const postForm = document.getElementById('forum-post-form');
        if (postForm) {
            postForm.addEventListener('submit', (e) => this.handlePostSubmit(e));
        }

        // Comment form
        const submitCommentBtn = document.getElementById('forum-submit-comment');
        if (submitCommentBtn) {
            submitCommentBtn.addEventListener('click', () => this.handleCommentSubmit());
        }

        // Image preview
        const imageInput = document.getElementById('forum-post-image');
        if (imageInput) {
            imageInput.addEventListener('change', (e) => this.handleImagePreview(e));
        }

        // User search
        const searchBtn = document.getElementById('forum-user-search-btn');
        const searchInput = document.getElementById('forum-user-search-input');
        if (searchBtn && searchInput) {
            searchBtn.addEventListener('click', () => this.handleUserSearch());
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.handleUserSearch();
            });
        }
    },

    loadCategories: async function () {
        try {
            const response = await fetch('/api/forum/categories');
            const data = await response.json();
            this.categories = data.categories;
            this.renderCategories();
        } catch (err) {
            console.error("[Forum] Failed to load categories:", err);
        }
    },

    renderCategories: function () {
        const listEl = document.getElementById('forum-categories-list');
        if (!listEl) return;

        listEl.innerHTML = this.categories.map(cat => `
            <div class="forum-cat-item" data-id="${cat.id}">
                <span class="forum-cat-name">${cat.name}</span>
                <span class="forum-cat-desc">${cat.description}</span>
            </div>
        `).join('');

        // Attach listeners
        listEl.querySelectorAll('.forum-cat-item').forEach(item => {
            item.addEventListener('click', () => {
                const catId = parseInt(item.getAttribute('data-id'));
                this.selectCategory(catId);
            });
        });
    },

    selectCategory: async function (catId) {
        this.currentCategoryId = catId;
        const category = this.categories.find(c => c.id === catId);

        // Update UI
        document.querySelectorAll('.forum-cat-item').forEach(item => {
            item.classList.toggle('active', parseInt(item.getAttribute('data-id')) === catId);
        });

        document.getElementById('forum-category-title').textContent = category.name;
        document.getElementById('forum-category-desc').textContent = category.description;

        // Show/hide New Post button based on guest status
        // restriction: guests cannot post
        const isGuest = window.currentUserIsGuest || (window.currentUser === null);
        let hideNewPost = isGuest;
        
        // Restriction: Only moderators can post in the News category
        if (category.name === "News" && !window.currentUserIsMod) {
            hideNewPost = true;
        }
        
        document.getElementById('forum-new-post-btn').classList.toggle('hidden', hideNewPost);

        await this.loadPosts(catId);
        this.showListView();
    },

    loadPosts: async function (catId) {
        const postsList = document.getElementById('forum-posts-list');
        postsList.innerHTML = '<div class="forum-placeholder"><h3>Loading posts...</h3></div>';

        try {
            const response = await fetch(`/api/forum/posts/${catId}`);
            const data = await response.json();
            this.renderPosts(data.posts);
        } catch (err) {
            console.error("[Forum] Failed to load posts:", err);
            postsList.innerHTML = '<div class="forum-placeholder"><h3>Error loading posts.</h3></div>';
        }
    },

    handleUserSearch: async function () {
        const username = document.getElementById('forum-user-search-input').value.trim();
        if (!username) return;

        console.log(`[Forum] Searching posts for user: ${username}`);

        // Clear active category
        document.querySelectorAll('.forum-cat-item').forEach(item => item.classList.remove('active'));
        this.currentCategoryId = null;

        // Update UI Header
        document.getElementById('forum-category-title').textContent = `Posts by ${username}`;
        document.getElementById('forum-category-desc').textContent = `Viewing all forum contributions from ${username}.`;
        document.getElementById('forum-new-post-btn').classList.add('hidden');

        const postsList = document.getElementById('forum-posts-list');
        postsList.innerHTML = '<div class="forum-placeholder"><h3>Searching...</h3></div>';

        try {
            const response = await fetch(`/api/forum/posts/user/${encodeURIComponent(username)}`);
            const data = await response.json();

            if (data.posts && data.posts.length > 0) {
                this.renderPosts(data.posts);
            } else {
                postsList.innerHTML = `
                    <div class="forum-placeholder">
                        <div class="placeholder-icon">🔍</div>
                        <h3>No posts found</h3>
                        <p>User "${username}" has not posted anything yet.</p>
                    </div>
                `;
            }
            this.showListView();
        } catch (err) {
            console.error("[Forum] User search error:", err);
            postsList.innerHTML = '<div class="forum-placeholder"><h3>Error performing search.</h3></div>';
        }
    },

    renderPosts: function (posts) {
        const postsList = document.getElementById('forum-posts-list');
        if (posts.length === 0) {
            postsList.innerHTML = `
                <div class="forum-placeholder">
                    <div class="placeholder-icon">📭</div>
                    <h3>No threads yet</h3>
                    <p>Be the first to start a conversation in this category!</p>
                </div>
            `;
            return;
        }

        postsList.innerHTML = posts.map(post => {
            const date = new Date(post.timestamp);
            const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            return `
                <div class="forum-post-card" data-id="${post.id}">
                    <div class="post-card-header">
                        <span class="post-card-title">${this.escapeHtml(post.title)}</span>
                        <span class="post-card-meta">
                            <span>Posted by <strong>${post.username}</strong></span>
                            <span>${dateStr}</span>
                        </span>
                    </div>
                    <div class="post-card-excerpt">${this.escapeHtml(post.content)}</div>
                    <div class="post-stats">
                        <div class="stat-item">💬 ${post.comment_count} comments</div>
                        ${post.image_url ? '<div class="stat-item">🖼️ Includes image</div>' : ''}
                    </div>
                </div>
            `;
        }).join('');

        // Attach listeners
        postsList.querySelectorAll('.forum-post-card').forEach(card => {
            card.addEventListener('click', () => {
                const postId = parseInt(card.getAttribute('data-id'));
                this.loadPostDetail(postId);
            });
        });
    },

    loadPostDetail: async function (postId) {
        this.currentPostId = postId;
        try {
            const response = await fetch(`/api/forum/post/${postId}`);
            const data = await response.json();
            this.renderPostDetail(data.post, data.comments);
            this.showPostView();
        } catch (err) {
            console.error("[Forum] Failed to load post detail:", err);
        }
    },

    renderPostDetail: function (post, comments) {
        const detailEl = document.getElementById('forum-post-detail');
        const date = new Date(post.timestamp);
        const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        detailEl.innerHTML = `
            <div class="post-detail-header">
                <h1 class="post-detail-title">${this.escapeHtml(post.title)}</h1>
                <div class="post-author-box">
                    <div class="author-avatar">${post.username[0].toUpperCase()}</div>
                    <div class="author-info">
                        <span class="author-name">${post.username} ${post.country_flag || ''}</span>
                        <span class="post-date">${dateStr}</span>
                    </div>
                </div>
            </div>
            <div class="post-content">${this.escapeHtml(post.content)}</div>
            ${post.image_url ? `
                <div class="post-image-container">
                    <img src="${post.image_url}" class="post-image" alt="Post attachment">
                </div>
            ` : ''}
        `;

        // Render comments
        const commentsListEl = document.getElementById('forum-comments-list');
        document.getElementById('forum-comment-count').textContent = `${comments.length} comments`;

        if (comments.length === 0) {
            commentsListEl.innerHTML = '<p class="forum-placeholder">No comments yet. Start the discussion!</p>';
        } else {
            commentsListEl.innerHTML = comments.map(c => {
                const cDate = new Date(c.timestamp).toLocaleString([], { hour: '2-digit', minute: '2-digit', month: 'short', day: 'numeric' });
                return `
                    <div class="forum-comment">
                        <div class="comment-avatar">${c.username[0].toUpperCase()}</div>
                        <div class="comment-body">
                            <div class="comment-header">
                                <span class="comment-author">${c.username}</span>
                                <span class="comment-date">${cDate}</span>
                            </div>
                            <div class="comment-content">${this.escapeHtml(c.content)}</div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        // Show/hide comment form
        const isGuest = window.currentUserIsGuest || (window.currentUser === null);
        document.getElementById('forum-comment-form-container').classList.toggle('hidden', isGuest);
    },

    handlePostSubmit: async function (e) {
        e.preventDefault();
        const title = document.getElementById('forum-post-title').value;
        const content = document.getElementById('forum-post-content').value;
        const catId = document.getElementById('forum-post-category-id').value;
        const imageFile = document.getElementById('forum-post-image').files[0];

        if (!title || !content) return;

        const formData = new FormData();
        formData.append('category_id', catId);
        formData.append('title', title);
        formData.append('content', content);
        if (imageFile) {
            formData.append('image', imageFile);
        }

        try {
            const response = await fetch('/api/forum/posts', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (data.success) {
                // Return to list and reload
                document.getElementById('forum-post-form').reset();
                this.handleImagePreview({ target: { files: [] } }); // Reset preview
                await this.selectCategory(this.currentCategoryId);
            } else {
                alert(data.error);
            }
        } catch (err) {
            console.error("[Forum] Post submit error:", err);
            alert("Failed to create post.");
        }
    },

    handleCommentSubmit: async function () {
        const content = document.getElementById('forum-comment-input').value;
        if (!content) return;

        try {
            const response = await fetch('/api/forum/comments', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    post_id: this.currentPostId,
                    content: content
                })
            });
            const data = await response.json();
            if (data.success) {
                document.getElementById('forum-comment-input').value = '';
                await this.loadPostDetail(this.currentPostId);
            } else {
                alert(data.error);
            }
        } catch (err) {
            console.error("[Forum] Comment submit error:", err);
            alert("Failed to post comment.");
        }
    },

    handleImagePreview: function (e) {
        const file = e.target.files[0];
        const previewEl = document.getElementById('forum-image-preview');

        if (file) {
            const reader = new FileReader();
            reader.onload = function (event) {
                previewEl.innerHTML = `<img src="${event.target.result}">`;
                previewEl.classList.remove('hidden');
            };
            reader.readAsDataURL(file);
        } else {
            previewEl.innerHTML = '';
            previewEl.classList.add('hidden');
        }
    },

    showListView: function () {
        document.querySelectorAll('.forum-view').forEach(v => v.classList.remove('active'));
        document.getElementById('forum-view-list').classList.add('active');
    },

    showPostView: function () {
        document.querySelectorAll('.forum-view').forEach(v => v.classList.remove('active'));
        document.getElementById('forum-view-post').classList.add('active');
    },

    showCreateView: function () {
        if (!this.currentCategoryId) return;

        document.querySelectorAll('.forum-view').forEach(v => v.classList.remove('active'));
        document.getElementById('forum-view-create').classList.add('active');
        document.getElementById('forum-post-category-id').value = this.currentCategoryId;

        // Show image upload for Screenshots, General, and Tips categories
        const category = this.categories.find(c => c.id === this.currentCategoryId);
        const allowUpload = ["Screenshots", "General", "Tips, Tricks, and Strategies", "Bugs/Errors", "News", "Suggestions"].includes(category.name);
        document.getElementById('forum-image-upload-section').classList.toggle('hidden', !allowUpload);
    },

    showRestrictedView: function () {
        document.querySelectorAll('.forum-view').forEach(v => v.classList.remove('active'));
        document.getElementById('forum-view-restricted').classList.add('active');
    },

    escapeHtml: function (text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
};

window.initForum = function () {
    Forum.init();
};
