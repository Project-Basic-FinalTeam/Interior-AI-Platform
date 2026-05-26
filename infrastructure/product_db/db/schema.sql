CREATE DATABASE IF NOT EXISTS product_db
DEFAULT CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE product_db;

DROP TABLE IF EXISTS product_images;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS categories;

CREATE TABLE categories (
    id INT AUTO_INCREMENT PRIMARY KEY,
    category_name VARCHAR(100) NOT NULL,
    category_code VARCHAR(100) NOT NULL UNIQUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE products (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    category_id INT NOT NULL,

    product_name VARCHAR(255) NOT NULL,
    price INT NOT NULL,

    width_x FLOAT NOT NULL,
    depth_y FLOAT NOT NULL,
    height_z FLOAT NOT NULL,
    size_unit VARCHAR(20) NOT NULL DEFAULT 'cm',

    product_url VARCHAR(700) NOT NULL,
    source_site VARCHAR(100) NOT NULL DEFAULT 'IKEA',

    raw_size_text TEXT NULL,
    raw_json JSON NULL,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    CONSTRAINT fk_products_category
        FOREIGN KEY (category_id)
        REFERENCES categories(id),

    CONSTRAINT uq_product_url
        UNIQUE (product_url)
);

CREATE TABLE product_images (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    product_id BIGINT NOT NULL,

    image_url VARCHAR(700) NULL,
    image_filename VARCHAR(255) NOT NULL,
    image_mime VARCHAR(100) NOT NULL,
    image_data LONGBLOB NOT NULL,
    image_description TEXT NULL,
    is_main BOOLEAN DEFAULT TRUE,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_product_images_product
        FOREIGN KEY (product_id)
        REFERENCES products(id)
        ON DELETE CASCADE
);

INSERT INTO categories (category_name, category_code)
VALUES
('책상', 'desk'),
('의자', 'chair'),
('선반', 'shelf')
ON DUPLICATE KEY UPDATE
category_name = VALUES(category_name);